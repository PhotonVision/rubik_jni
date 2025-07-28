/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * NOTE INTELLISENSE WILL NOT WORK UNTIL THE PROJECT IS BUILT AT LEAST ONCE
 */

#include <jni.h>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/version.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

typedef struct _BOX_RECT {
  int left;
  int right;
  int top;
  int bottom;
} BOX_RECT;

typedef struct __detect_result_t {
  int id;
  BOX_RECT box;
  float obj_conf;
} detect_result_t;

// JNI class reference (this can be global since it's shared)
static jclass detectionResultClass = nullptr;
static jclass runtimeExceptionClass = nullptr;

// Guesses the width, height, and channels of a tensor if it were an image.
// Returns false on failure.
bool tensor_image_dims(const TfLiteTensor *tensor, int *w, int *h, int *c) {
  int n = TfLiteTensorNumDims(tensor);
  int cursor = 0;

  for (int i = 0; i < n; i++) {
    int dim = TfLiteTensorDim(tensor, i);
    if (dim == 0)
      return false;
    if (dim == 1)
      continue;

    switch (cursor++) {
    case 0:
      if (w)
        *w = dim;
      break;
    case 1:
      if (h)
        *h = dim;
      break;
    case 2:
      if (c)
        *c = dim;
      break;
    default:
      return false;
      break;
    }
  }

  // Ensure that we at least have the width and height.
  if (cursor < 2)
    return false;
  // If we don't have the number of channels, then assume there's only one.
  if (cursor == 2 && c)
    *c = 1;
  // Ensure we have no more than 4 image channels.
  if (*c > 4)
    return false;
  // The tensor dimension appears coherent.
  return true;
}

void print_tensor_info(const TfLiteTensor *tensor) {
  size_t tensor_size = TfLiteTensorByteSize(tensor);

  std::printf("INFO:   Size: %lu bytes\n", tensor_size);

  int num_dims = TfLiteTensorNumDims(tensor);

  std::printf("INFO:   Dimension: ");

  for (int i = 0; i < num_dims; i++)
    std::printf("%d%s", TfLiteTensorDim(tensor, i),
                i == num_dims - 1 ? "" : "x");

  std::printf("\n");

  switch (TfLiteTensorType(tensor)) {
  case kTfLiteFloat16:
    std::printf("INFO:   Type: f16\n");
    break;
  case kTfLiteFloat32:
    std::printf("INFO:   Type: f32\n");
    break;
  case kTfLiteUInt8:
    std::printf("INFO:   Type: u8 \n");
    break;
  case kTfLiteUInt32:
    std::printf("INFO:   Type: u32\n");
    break;
  case kTfLiteInt8:
    std::printf("INFO:   Type: i8 \n");
    break;
  case kTfLiteInt32:
    std::printf("INFO:   Type: i32\n");
    break;
  default:
    std::printf("INFO:   Type: ???\n");
    break;
  }
}

extern "C" {
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // Cache exception classes
  jclass localRuntimeClass = env->FindClass("java/lang/RuntimeException");
  if (localRuntimeClass) {
    runtimeExceptionClass = (jclass)env->NewGlobalRef(localRuntimeClass);
    env->DeleteLocalRef(localRuntimeClass);
  }

  // Find the detection result class
  jclass localClass =
      env->FindClass("org/photonvision/rubik/RubikJNI$RubikResult");
  if (!localClass) {
    std::printf(
        "Couldn't find class org/photonvision/rubik/RubikJNI$RubikResult!\n");
    return JNI_ERR;
  }

  // Create global reference
  detectionResultClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);

  if (!detectionResultClass) {
    std::printf("Couldn't create global reference to class!\n");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
    if (detectionResultClass) {
      env->DeleteGlobalRef(detectionResultClass);
      detectionResultClass = nullptr;
    }
  }
}

static jobject MakeJObject(JNIEnv *env, const detect_result_t &result) {
  if (!detectionResultClass) {
    std::cerr << "ERROR: detectionResultClass is null" << std::endl;
    return nullptr;
  }

  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");
  if (!constructor) {
    std::cerr << "ERROR: Could not find constructor for RknnResult"
              << std::endl;
    return nullptr;
  }

  return env->NewObject(detectionResultClass, constructor, result.box.left,
                        result.box.top, result.box.right, result.box.bottom,
                        result.obj_conf, result.id);
}

// Helper function to throw exceptions
void ThrowRuntimeException(JNIEnv *env, const char *message) {
  if (runtimeExceptionClass) {
    env->ThrowNew(runtimeExceptionClass, message);
  }
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    create
 * Signature: (Ljava/lang/String;)?
 */
JNIEXPORT jarray JNICALL
Java_org_photonvision_rubik_RubikJNI_create
  (JNIEnv *env, jobject obj, jstring modelPath)
{
  const char *model_name = env->GetStringUTFChars(modelPath, nullptr);
  if (model_name == nullptr) {
    std::cerr << "ERROR: Failed to retrieve model path from Java string."
              << std::endl;
    ThrowRuntimeException(env, "Failed to retrieve model path");
    return 0;
  }

  // Load the model
  TfLiteModel *model = TfLiteModelCreateFromFile(model_name);
  if (!model) {
    std::printf("ERROR: Failed to load model file '%s'\n", model_name);
    ThrowRuntimeException(env, "Failed to load model file");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  std::printf("INFO: Loaded model file '%s'\n", model_name);

  // Create external delegate options
  auto delegateOptsValue =
      TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");
  TfLiteExternalDelegateOptions *delegateOpts = &delegateOptsValue;
  if (!delegateOpts) {
    std::printf("ERROR: Failed to create delegate options\n");
    ThrowRuntimeException(env, "Failed to create delegate options");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  TfLiteExternalDelegateOptionsInsert(delegateOpts, "backend_type", "htp");

  // Create the delegate
  TfLiteDelegate *delegate = TfLiteExternalDelegateCreate(delegateOpts);

  if (!delegate) {
    std::printf("ERROR: Failed to create external delegate\n");
    ThrowRuntimeException(env, "Failed to create external delegate");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  std::printf("INFO: Loaded external delegate\n");

  // Create interpreter options
  TfLiteInterpreterOptions *interpreterOpts = TfLiteInterpreterOptionsCreate();
  if (!interpreterOpts) {
    std::printf("ERROR: Failed to create interpreter options\n");
    ThrowRuntimeException(env, "Failed to create interpreter options");
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  TfLiteInterpreterOptionsAddDelegate(interpreterOpts, delegate);

  // Create the interpreter
  TfLiteInterpreter *interpreter =
      TfLiteInterpreterCreate(model, interpreterOpts);
  TfLiteInterpreterOptionsDelete(interpreterOpts);

  if (!interpreter) {
    std::printf("ERROR: Failed to create interpreter\n");
    ThrowRuntimeException(env, "Failed to create interpreter");
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  // Modify graph with delegate
  if (TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate) !=
      kTfLiteOk) {
    std::printf("ERROR: Failed to modify graph with delegate\n");
    ThrowRuntimeException(env, "Failed to modify graph with delegate");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  // Allocate tensors
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    std::printf("ERROR: Failed to allocate tensors\n");
    ThrowRuntimeException(env, "Failed to allocate tensors");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  env->ReleaseStringUTFChars(modelPath, model_name);

  jlongArray ptrs = env->NewLongArray(3);
  if (!ptrs) {
    std::cerr << "ERROR: Failed to create jlongArray" << std::endl;
    ThrowRuntimeException(env, "Failed to create jlongArray");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    TfLiteModelDelete(model);
    return nullptr;
  }

  jlong values[3];
  values[0] = reinterpret_cast<jlong>(interpreter);
  values[1] = reinterpret_cast<jlong>(delegate);
  values[2] = reinterpret_cast<jlong>(model);

  env->SetLongArrayRegion(ptrs, 0, 3, values);

  std::printf("INFO: TensorFlow Lite initialization completed successfully\n");

  return ptrs;
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    destroy
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_rubik_RubikJNI_destroy
  (JNIEnv *env, jclass, jlongArray ptrs)
{
  TfLiteInterpreter *interpreter = reinterpret_cast<TfLiteInterpreter *>(
      env->GetLongArrayElements(ptrs, nullptr)[0]);
  if (!interpreter) {
    std::cerr << "ERROR: Invalid interpreter handle" << std::endl;
    ThrowRuntimeException(env, "Invalid interpreter handle");
    return;
  }

  TfLiteDelegate *delegate = reinterpret_cast<TfLiteDelegate *>(
      env->GetLongArrayElements(ptrs, nullptr)[1]);
  if (!delegate) {
    std::cerr << "ERROR: Invalid delegate handle" << std::endl;
    ThrowRuntimeException(env, "Invalid delegate handle");
    return;
  }

  TfLiteModel *model = reinterpret_cast<TfLiteModel *>(
      env->GetLongArrayElements(ptrs, nullptr)[2]);
  if (!model) {
    std::cerr << "ERROR: Invalid model handle" << std::endl;
    ThrowRuntimeException(env, "Invalid model handle");
    return;
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteExternalDelegateDelete(delegate);
  TfLiteModelDelete(model);

  std::printf("INFO: Object Detection instance destroyed successfully\n");
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    detect
 * Signature: (JJD)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_rubik_RubikJNI_detect
  (JNIEnv *env, jobject obj, jlong interpreterPtr, jlong input_cvmat_ptr,
   jdouble boxThresh)
{
  // Check if the interpreter pointer is valid
  TfLiteInterpreter *interpreter =
      reinterpret_cast<TfLiteInterpreter *>(interpreterPtr);

  if (!interpreter) {
    std::cerr << "ERROR: Invalid interpreter handle" << std::endl;
    ThrowRuntimeException(env, "Invalid interpreter handle");
    return nullptr;
  }

  /* Get the tensor info. */

  // Input:

  unsigned int n_tensors_in = TfLiteInterpreterGetInputTensorCount(interpreter);

  if (n_tensors_in != 1) {
    std::cerr << "ERROR: expected only 1 input tensor, got " << n_tensors_in
              << std::endl;
    ThrowRuntimeException(env, "Expected only 1 input tensor");
    return nullptr;
  }

  TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);

  std::printf("INFO: Input tensor:\n");
  print_tensor_info(input);

  int in_w, in_h, in_c;

  if (!tensor_image_dims(input, &in_w, &in_h, &in_c)) {
    std::cerr << "ERROR: failed to extract image dimensions of input tensor."
              << std::endl;
    ThrowRuntimeException(env,
                          "Failed to extract image dimensions of input tensor");
    return nullptr;
  }

  std::printf("INFO: input tensor image dimensions: %dx%d, with %d channels\n",
              in_w, in_h, in_c);

  // Output:

  unsigned int n_tensors_out =
      TfLiteInterpreterGetOutputTensorCount(interpreter);

  if (n_tensors_out != 3) {
    std::cerr << "ERROR: expected 3 output tensors, got " << n_tensors_out
              << std::endl;
    ThrowRuntimeException(env, "Expected 3 output tensors");
    return nullptr;
  }

  const TfLiteTensor *output = TfLiteInterpreterGetOutputTensor(interpreter, 0);

  std::printf("INFO: Output tensor:\n");
  print_tensor_info(output);

  int out_w, out_h, out_c;

  if (!tensor_image_dims(output, &out_w, &out_h, &out_c)) {
    std::cerr << "ERROR: failed to extract image dimensions of output tensor."
              << std::endl;
    ThrowRuntimeException(
        env, "Failed to extract image dimensions of output tensor");
    return nullptr;
  }

  std::printf("INFO: output tensor image dimensions: %dx%d, with %d channels\n",
              out_w, out_h, out_c);

  /* Load the input. */

  cv::Mat *input_img = reinterpret_cast<cv::Mat *>(input_cvmat_ptr);
  if (!input_img || input_img->empty()) {
    std::cerr << "ERROR: Invalid input image" << std::endl;
    ThrowRuntimeException(env, "Invalid input image");
    return nullptr;
  }

  if (in_w != input_img->cols || in_h != input_img->rows) {
    std::cerr << "ERROR: input image dimensions (" << input_img->cols << "x"
              << input_img->rows << ") do not match input tensor dimensions ("
              << in_w << "x" << in_h << ")." << std::endl;
    ThrowRuntimeException(
        env, "Input image dimensions do not match input tensor dimensions");
    return nullptr;
  }

  // If the dimension and channels match, the byte size of the input and the
  // tensor should be identical.
  size_t image_size = input_img->cols * input_img->rows * in_c;
  size_t tensor_size = TfLiteTensorByteSize(input);
  assert(tensor_size == image_size);

  // Write the input data into the tensor.
  std::memcpy(TfLiteTensorData(input), input_img->data, tensor_size);

  // Run inference
  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    std::cerr << "ERROR: Failed to invoke interpreter" << std::endl;
    ThrowRuntimeException(env, "Failed to invoke interpreter");
    return nullptr;
  }

  const TfLiteTensor *outputTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  if (!outputTensor) {
    std::cerr << "ERROR: Failed to get output tensor" << std::endl;
    ThrowRuntimeException(env, "Failed to get output tensor");
    return nullptr;
  }

  std::printf("INFO: Output tensor:\n");
  print_tensor_info(outputTensor);

  // Get tensor dimensions and validate structure
  int batchSize = TfLiteTensorDim(outputTensor, 0);     // Should be 1
  int numDetections = TfLiteTensorDim(outputTensor, 1); // Should be 8400
  int boxCoords = TfLiteTensorDim(outputTensor, 2);     // Should be 4

  std::printf("INFO: Tensor dimensions: batch=%d, detections=%d, coords=%d\n",
              batchSize, numDetections, boxCoords);

  if (batchSize != 1 || boxCoords != 4) {
    std::cerr << "ERROR: Unexpected tensor dimensions" << std::endl;
    ThrowRuntimeException(env, "Unexpected tensor dimensions");
    return nullptr;
  }

  // Get the other output tensors
  const TfLiteTensor *scoresTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 1);
  const TfLiteTensor *classesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 2);

  if (!scoresTensor || !classesTensor) {
    std::cerr << "ERROR: Failed to get scores or classes tensor" << std::endl;
    ThrowRuntimeException(env, "Failed to get scores or classes tensor");
    return nullptr;
  }

  // Get tensor data - handle both float32 and quantized types
  bool isQuantized = (TfLiteTensorType(outputTensor) == kTfLiteUInt8);
  std::printf("INFO: Model is %s\n", isQuantized ? "quantized" : "float32");

  std::vector<detect_result_t> results;
  std::vector<detect_result_t> candidateResults; // For NMS

  // Scale factors for coordinate conversion
  float scaleX = static_cast<float>(input_img->cols);
  float scaleY = static_cast<float>(input_img->rows);

  // Determine number of classes from scores tensor
  int numClasses = TfLiteTensorDim(scoresTensor, 1);
  if (TfLiteTensorNumDims(scoresTensor) == 3) {
    numClasses = TfLiteTensorDim(scoresTensor, 2);
  }
  std::printf("INFO: Number of classes: %d\n", numClasses);

  // Process detections based on tensor type
  if (isQuantized) {
    // Handle quantized tensors
    uint8_t *boxesData = static_cast<uint8_t *>(TfLiteTensorData(outputTensor));
    uint8_t *scoresData =
        static_cast<uint8_t *>(TfLiteTensorData(scoresTensor));
    uint8_t *classesData =
        static_cast<uint8_t *>(TfLiteTensorData(classesTensor));

    // Get quantization parameters
    TfLiteQuantizationParams boxesParams =
        TfLiteTensorQuantizationParams(outputTensor);
    TfLiteQuantizationParams scoresParams =
        TfLiteTensorQuantizationParams(scoresTensor);
    TfLiteQuantizationParams classesParams =
        TfLiteTensorQuantizationParams(classesTensor);

    std::printf("INFO: Quantization - Boxes: scale=%f, zero=%d | Scores: "
                "scale=%f, zero=%d\n",
                boxesParams.scale, boxesParams.zero_point, scoresParams.scale,
                scoresParams.zero_point);

    for (int i = 0; i < numDetections; i++) {
      // Dequantize bounding boxes (assuming normalized YOLO format: x_center,
      // y_center, width, height)
      float x_center =
          (boxesData[i * 4 + 0] - boxesParams.zero_point) * boxesParams.scale;
      float y_center =
          (boxesData[i * 4 + 1] - boxesParams.zero_point) * boxesParams.scale;
      float width =
          (boxesData[i * 4 + 2] - boxesParams.zero_point) * boxesParams.scale;
      float height =
          (boxesData[i * 4 + 3] - boxesParams.zero_point) * boxesParams.scale;

      // Convert from center-width-height to corner coordinates
      float x1 = x_center - width / 2.0f;
      float y1 = y_center - height / 2.0f;
      float x2 = x_center + width / 2.0f;
      float y2 = y_center + height / 2.0f;

      // Find best class and score
      float maxScore = 0.0f;
      int bestClass = -1;

      if (numClasses == 1) {
        // Binary classification
        maxScore =
            (scoresData[i] - scoresParams.zero_point) * scoresParams.scale;
        bestClass = 0;
      } else {
        // Multi-class: find maximum score across all classes
        for (int c = 0; c < numClasses; c++) {
          int scoreIndex = i * numClasses + c;
          float score = (scoresData[scoreIndex] - scoresParams.zero_point) *
                        scoresParams.scale;
          if (score > maxScore) {
            maxScore = score;
            bestClass = c;
          }
        }
      }

      // Apply confidence threshold
      if (maxScore < boxThresh) {
        continue;
      }

      // Convert normalized coordinates to pixel coordinates
      int left = static_cast<int>(x1 * scaleX);
      int top = static_cast<int>(y1 * scaleY);
      int right = static_cast<int>(x2 * scaleX);
      int bottom = static_cast<int>(y2 * scaleY);

      // Clamp to image boundaries
      left = std::max(0, std::min(left, input_img->cols - 1));
      top = std::max(0, std::min(top, input_img->rows - 1));
      right = std::max(left + 1, std::min(right, input_img->cols));
      bottom = std::max(top + 1, std::min(bottom, input_img->rows));

      // Create detection result
      detect_result_t detection;
      detection.box.left = left;
      detection.box.top = top;
      detection.box.right = right;
      detection.box.bottom = bottom;
      detection.obj_conf = maxScore;
      detection.id = bestClass;

      candidateResults.push_back(detection);
    }
  } else {
    // Handle float32 tensors
    float *boxesData = static_cast<float *>(TfLiteTensorData(outputTensor));
    float *scoresData = static_cast<float *>(TfLiteTensorData(scoresTensor));
    float *classesData = static_cast<float *>(TfLiteTensorData(classesTensor));

    for (int i = 0; i < numDetections; i++) {
      // Extract bounding boxes (assuming YOLO format)
      float x_center = boxesData[i * 4 + 0];
      float y_center = boxesData[i * 4 + 1];
      float width = boxesData[i * 4 + 2];
      float height = boxesData[i * 4 + 3];

      // Convert to corner coordinates
      float x1 = x_center - width / 2.0f;
      float y1 = y_center - height / 2.0f;
      float x2 = x_center + width / 2.0f;
      float y2 = y_center + height / 2.0f;

      // Find best class and score
      float maxScore = 0.0f;
      int bestClass = -1;

      if (numClasses == 1) {
        maxScore = scoresData[i];
        bestClass = 0;
      } else {
        for (int c = 0; c < numClasses; c++) {
          int scoreIndex = i * numClasses + c;
          float score = scoresData[scoreIndex];
          if (score > maxScore) {
            maxScore = score;
            bestClass = c;
          }
        }
      }

      if (maxScore < boxThresh) {
        continue;
      }

      // Convert to pixel coordinates
      int left = static_cast<int>(x1 * scaleX);
      int top = static_cast<int>(y1 * scaleY);
      int right = static_cast<int>(x2 * scaleX);
      int bottom = static_cast<int>(y2 * scaleY);

      // Clamp to image boundaries
      left = std::max(0, std::min(left, input_img->cols - 1));
      top = std::max(0, std::min(top, input_img->rows - 1));
      right = std::max(left + 1, std::min(right, input_img->cols));
      bottom = std::max(top + 1, std::min(bottom, input_img->rows));

      detect_result_t detection;
      detection.box.left = left;
      detection.box.top = top;
      detection.box.right = right;
      detection.box.bottom = bottom;
      detection.obj_conf = maxScore;
      detection.id = bestClass;

      candidateResults.push_back(detection);
    }
  }

  // Apply Non-Maximum Suppression (NMS)
  std::sort(candidateResults.begin(), candidateResults.end(),
            [](const detect_result_t &a, const detect_result_t &b) {
              return a.obj_conf > b.obj_conf;
            });

  const float nmsThreshold = 0.4f; // IoU threshold for NMS

  for (size_t i = 0; i < candidateResults.size(); i++) {
    if (candidateResults[i].obj_conf == 0.0f)
      continue; // Already suppressed

    results.push_back(candidateResults[i]);

    // Suppress overlapping detections
    for (size_t j = i + 1; j < candidateResults.size(); j++) {
      if (candidateResults[j].obj_conf == 0.0f)
        continue;
      if (candidateResults[i].id != candidateResults[j].id)
        continue; // Different classes

      // Calculate IoU
      int x1 =
          std::max(candidateResults[i].box.left, candidateResults[j].box.left);
      int y1 =
          std::max(candidateResults[i].box.top, candidateResults[j].box.top);
      int x2 = std::min(candidateResults[i].box.right,
                        candidateResults[j].box.right);
      int y2 = std::min(candidateResults[i].box.bottom,
                        candidateResults[j].box.bottom);

      if (x2 <= x1 || y2 <= y1)
        continue; // No overlap

      int intersectionArea = (x2 - x1) * (y2 - y1);
      int area1 =
          (candidateResults[i].box.right - candidateResults[i].box.left) *
          (candidateResults[i].box.bottom - candidateResults[i].box.top);
      int area2 =
          (candidateResults[j].box.right - candidateResults[j].box.left) *
          (candidateResults[j].box.bottom - candidateResults[j].box.top);
      int unionArea = area1 + area2 - intersectionArea;

      float iou =
          static_cast<float>(intersectionArea) / static_cast<float>(unionArea);

      if (iou > nmsThreshold) {
        candidateResults[j].obj_conf = 0.0f; // Suppress this detection
      }
    }
  }

  std::printf("INFO: After NMS: %zu detections from %zu candidates\n",
              results.size(), candidateResults.size());
}
} // extern "C"
