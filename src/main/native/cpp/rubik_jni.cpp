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
  // Get the array elements ONCE
  jlong *elements = env->GetLongArrayElements(ptrs, nullptr);
  if (!elements) {
    ThrowRuntimeException(env, "Failed to get array elements");
    return;
  }

  // Extract all pointers from the single array
  TfLiteInterpreter *interpreter =
      reinterpret_cast<TfLiteInterpreter *>(elements[0]);
  TfLiteDelegate *delegate = reinterpret_cast<TfLiteDelegate *>(elements[1]);
  TfLiteModel *model = reinterpret_cast<TfLiteModel *>(elements[2]);

  // Release the array back to the JVM (CRITICAL!)
  env->ReleaseLongArrayElements(ptrs, elements, 0);

  // Now safely use the pointers
  if (interpreter)
    TfLiteInterpreterDelete(interpreter);
  if (delegate)
    TfLiteExternalDelegateDelete(delegate);
  if (model)
    TfLiteModelDelete(model);

  std::printf("INFO: Object Detection instance destroyed successfully\n");
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    detect
 * Signature: (JJDD)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_rubik_RubikJNI_detect
  (JNIEnv *env, jobject obj, jlong interpreterPtr, jlong input_cvmat_ptr,
   jdouble boxThresh, jdouble nmsThreshold)
{
  TfLiteInterpreter *interpreter =
      reinterpret_cast<TfLiteInterpreter *>(interpreterPtr);
  if (!interpreter) {
    ThrowRuntimeException(env, "Invalid interpreter handle");
    return nullptr;
  }

  TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);
  int in_w, in_h, in_c;
  if (!tensor_image_dims(input, &in_w, &in_h, &in_c)) {
    ThrowRuntimeException(env, "Invalid input tensor shape");
    return nullptr;
  }

  cv::Mat *input_img = reinterpret_cast<cv::Mat *>(input_cvmat_ptr);
  if (!input_img || input_img->empty() || input_img->cols != in_w ||
      input_img->rows != in_h) {
    ThrowRuntimeException(env, "Invalid input image or mismatched dimensions");
    return nullptr;
  }

  cv::Mat rgb;
  cv::cvtColor(*input_img, rgb, cv::COLOR_BGR2RGB);

  std::memcpy(TfLiteTensorData(input), rgb.data, TfLiteTensorByteSize(input));

  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    ThrowRuntimeException(env, "Interpreter invocation failed");
    return nullptr;
  }

  const TfLiteTensor *boxesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  const TfLiteTensor *scoresTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 1);
  const TfLiteTensor *classesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 2);

  const TfLiteQuantizationParams boxesParams =
      TfLiteTensorQuantizationParams(boxesTensor);
  const TfLiteQuantizationParams scoresParams =
      TfLiteTensorQuantizationParams(scoresTensor);

  const int numBoxes = TfLiteTensorDim(boxesTensor, 1);
  std::printf("INFO: Detected %d boxes\n", numBoxes);

  if (TfLiteTensorType(boxesTensor) != kTfLiteUInt8) {
    ThrowRuntimeException(env, "Expected uint8 tensor type");
    return nullptr;
  }

  if (TfLiteTensorType(scoresTensor) != kTfLiteUInt8) {
    ThrowRuntimeException(env, "Expected uint8 tensor type");
    return nullptr;
  }

  if (TfLiteTensorType(classesTensor) != kTfLiteUInt8) {
    ThrowRuntimeException(env, "Expected uint8 tensor type");
    return nullptr;
  }

  uint8_t *boxesData = static_cast<uint8_t *>(TfLiteTensorData(boxesTensor));
  uint8_t *scoresData = static_cast<uint8_t *>(TfLiteTensorData(scoresTensor));
  uint8_t *classesData =
      static_cast<uint8_t *>(TfLiteTensorData(classesTensor));

  std::printf("INFO: Boxes: [");
  for (int i = 0; i < numBoxes * 4; ++i) {
    std::printf("%d%s", boxesData[i], i == numBoxes * 4 - 1 ? "" : ", ");
  }
  std::printf("]\n");
  std::printf("INFO: Scores tensor: %p\n", static_cast<void *>(scoresData));
  std::printf("INFO: Classes tensor: %p\n", static_cast<void *>(classesData));

  std::printf("boxesTensor dims: [");
  for (int i = 0; i < TfLiteTensorNumDims(boxesTensor); ++i) {
    std::printf("%d%s", TfLiteTensorDim(boxesTensor, i),
                i == TfLiteTensorNumDims(boxesTensor) - 1 ? "]\n" : ", ");
  }

  std::printf("boxes scale: %f, zero_point: %d\n", boxesParams.scale,
              boxesParams.zero_point);

  std::vector<detect_result_t> candidateResults;

  // float scaleX = static_cast<float>(input_img->cols);
  // float scaleY = static_cast<float>(input_img->rows);

  for (int i = 0; i < numBoxes; ++i) {
    // DeQuantize
float x_center = ((boxesData[i * 4 + 0] - boxesParams.zero_point) * boxesParams.scale);
float y_center = ((boxesData[i * 4 + 1] - boxesParams.zero_point) * boxesParams.scale);
float width    = ((boxesData[i * 4 + 2] - boxesParams.zero_point) * boxesParams.scale);
float height   = ((boxesData[i * 4 + 3] - boxesParams.zero_point) * boxesParams.scale);


    float score =
        (scoresData[i] - scoresParams.zero_point) * scoresParams.scale;
    if (score < boxThresh)
      continue;

    int classId = classesData[i];

    float x1 = x_center - width / 2.0f;
    float y1 = y_center - height / 2.0f;
    float x2 = x_center + width / 2.0f;
    float y2 = y_center + height / 2.0f;

    x1 = std::max(0.0f, std::min(x1, static_cast<float>(input_img->cols - 1)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(input_img->rows - 1)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(input_img->cols - 1)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(input_img->rows - 1)));

      std::printf("DEBUG Box %d: raw=[%d,%d,%d,%d], "
                  "dequant=[%.3f,%.3f,%.3f,%.3f], score=%.3f\n",
                  i, boxesData[i * 4], boxesData[i * 4 + 1],
                  boxesData[i * 4 + 2], boxesData[i * 4 + 3], x_center,
                  y_center, width, height, score);
    
    detect_result_t det;
    det.box.left = x1;
    det.box.top = y1;
    det.box.right = x2;
    det.box.bottom = y2;
    det.obj_conf = score;
    det.id = classId;

    candidateResults.push_back(det);
    // std::printf(
    //     "INFO: Detected object %d: [%d, %d, %d, %d] with confidence %.2f\n",
    //     classId, det.box.left, det.box.top, det.box.right, det.box.bottom,
    //     det.obj_conf);
  }

  // NMS
  std::sort(candidateResults.begin(), candidateResults.end(),
            [](const detect_result_t &a, const detect_result_t &b) {
              return a.obj_conf > b.obj_conf;
            });

  std::vector<detect_result_t> results;

  for (size_t i = 0; i < candidateResults.size(); ++i) {
    if (candidateResults[i].obj_conf == 0.0f)
      continue;

    results.push_back(candidateResults[i]);

    for (size_t j = i + 1; j < candidateResults.size(); ++j) {
      if (candidateResults[j].obj_conf == 0.0f ||
          candidateResults[i].id != candidateResults[j].id)
        continue;

      int x1 =
          std::max(candidateResults[i].box.left, candidateResults[j].box.left);
      int y1 =
          std::max(candidateResults[i].box.top, candidateResults[j].box.top);
      int x2 = std::min(candidateResults[i].box.right,
                        candidateResults[j].box.right);
      int y2 = std::min(candidateResults[i].box.bottom,
                        candidateResults[j].box.bottom);

      if (x2 <= x1 || y2 <= y1)
        continue;

      int interArea = (x2 - x1) * (y2 - y1);
      int area1 =
          (candidateResults[i].box.right - candidateResults[i].box.left) *
          (candidateResults[i].box.bottom - candidateResults[i].box.top);
      int area2 =
          (candidateResults[j].box.right - candidateResults[j].box.left) *
          (candidateResults[j].box.bottom - candidateResults[j].box.top);
      float iou = static_cast<float>(interArea) / (area1 + area2 - interArea);

      if (iou > nmsThreshold)
        candidateResults[j].obj_conf = 0.0f;
    }
  }

  jobjectArray jResults =
      env->NewObjectArray(results.size(), detectionResultClass, nullptr);
  for (size_t i = 0; i < results.size(); ++i) {
    jobject jDet = MakeJObject(env, results[i]);
    env->SetObjectArrayElement(jResults, i, jDet);
  }

  std::printf("INFO: Returned %zu results after NMS\n", results.size());
  return jResults;
}

} // extern "C"
