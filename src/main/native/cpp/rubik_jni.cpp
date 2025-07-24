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
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
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

  jlong interpreterPtr = reinterpret_cast<jlong>(interpreter);

  std::printf("INFO: TensorFlow Lite initialization completed successfully "
              "(handle: %ld)\n",
              interpreterPtr);

  return interpreterPtr;
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_rubik_RubikJNI_destroy
  (JNIEnv *env, jclass, jlong interpreterPtr)
{
  TfLiteInterpreter *interpreter =
      reinterpret_cast<TfLiteInterpreter *>(interpreterPtr);
  if (!interpreter) {
    std::cerr << "ERROR: Invalid interpreter handle" << std::endl;
    ThrowRuntimeException(env, "Invalid interpreter handle");
    return;
  }

  // Delete the interpreter
  TfLiteInterpreterDelete(interpreter);

  std::printf("INFO: TensorFlow Lite interpreter destroyed successfully\n");
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

  cv::Mat *input_img = reinterpret_cast<cv::Mat *>(input_cvmat_ptr);
  if (!input_img || input_img->empty()) {
    std::cerr << "ERROR: Invalid input image" << std::endl;
    ThrowRuntimeException(env, "Invalid input image");
    return nullptr;
  }

  // Get input tensor

  int inputTensorIndex = TfLiteInterpreterGetInputTensorIndex(interpreter, 0);

  if (inputTensorIndex < 0) {
    std::cerr << "ERROR: Failed to get input tensor index" << std::endl;
    ThrowRuntimeException(env, "Failed to get input tensor index");
    return nullptr;
  }

  TfLiteTensor *inputTensor =
      TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
  if (!inputTensor) {
    std::cerr << "ERROR: Failed to get input tensor" << std::endl;
    ThrowRuntimeException(env, "Failed to get input tensor");
    return nullptr;
  }

  // Check input tensor type
  if (TfLiteTensorType(inputTensor) != kTfLiteUInt8) {
    std::cerr << "ERROR: Input tensor is not of type kTfLiteUInt8" << std::endl;
    ThrowRuntimeException(env, "Input tensor is not of type kTfLiteUInt8");
    return nullptr;
  }

  // Get input tensor dimensions
  int inputDims = TfLiteTensorNumDims(inputTensor);
  if (inputDims < 3) {
    std::cerr << "ERROR: Input tensor does not have enough dimensions"
              << std::endl;
    ThrowRuntimeException(env, "Input tensor does not have enough dimensions");
    return nullptr;
  }

  int inputHeight = TfLiteTensorDim(inputTensor, 1);
  int inputWidth = TfLiteTensorDim(inputTensor, 2);
  int inputChannels = TfLiteTensorDim(inputTensor, 3);

  // Check if input image matches expected dimensions
  if (input_img->rows != inputHeight || input_img->cols != inputWidth ||
      input_img->channels() != inputChannels) {
    std::cerr << "ERROR: Input image dimensions do not match expected input "
                 "tensor dimensions"
              << std::endl;
    ThrowRuntimeException(
        env,
        "Input image dimensions do not match expected input tensor dimensions");
    return nullptr;
  }

  // Copy image data to input tensor
  uint8_t *inputData = static_cast<uint8_t *>(TfLiteTensorData(inputTensor));
  if (!inputData) {
    std::cerr << "ERROR: Failed to get input tensor data pointer" << std::endl;
    ThrowRuntimeException(env, "Failed to get input tensor data pointer");
    return nullptr;
  }

  // Ensure input image is in the correct format (e.g., BGR to RGB if needed)

  cv::Mat rgbImage;
  if (inputChannels == 3) {
    // Convert BGR to RGB if necessary
    cv::cvtColor(*input_img, rgbImage, cv::COLOR_BGR2RGB);
  } else {
    // Assume input image is already in the correct format
    rgbImage = *input_img;
  }

  // TODO: Commenting out resizing for the time being, as it should be
  // letterboxed. Leaving it in place as it might prove valuable in the future.
  // Resize image to match input tensor dimensions
  // cv::Mat resizedImage;
  // cv::resize(rgbImage, resizedImage, cv::Size(inputWidth, inputHeight));
  // Copy resized image data to input tensor
  if (rgbImage.isContinuous()) {
    std::memcpy(inputData, rgbImage.data,
                rgbImage.total() * rgbImage.elemSize());
  } else {
    for (int i = 0; i < rgbImage.rows; ++i) {
      std::memcpy(inputData + i * rgbImage.step[0], rgbImage.ptr(i),
                  rgbImage.cols * rgbImage.elemSize());
    }
  }

  // Run inference
  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    std::cerr << "ERROR: Failed to invoke interpreter" << std::endl;
    ThrowRuntimeException(env, "Failed to invoke interpreter");
    return nullptr;
  }

  // Get output tensor

  int outputTensorIndex = TfLiteInterpreterGetOutputTensorIndex(interpreter, 0);
  if (outputTensorIndex < 0) {
    std::cerr << "ERROR: Failed to get output tensor index" << std::endl;
    ThrowRuntimeException(env, "Failed to get output tensor index");
    return nullptr;
  }

  const TfLiteTensor *outputTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
  if (!outputTensor) {
    std::cerr << "ERROR: Failed to get output tensor" << std::endl;
    ThrowRuntimeException(env, "Failed to get output tensor");
    return nullptr;
  }

  // Check output tensor type
  if (TfLiteTensorType(outputTensor) != kTfLiteUInt8) {
    std::cerr << "ERROR: Output tensor is not of type kTfLiteUInt8"
              << std::endl;
    ThrowRuntimeException(env, "Output tensor is not of type kTfLiteUInt8");
    return nullptr;
  }

  // Get output tensor dimensions
  int outputDims = TfLiteTensorNumDims(outputTensor);
  if (outputDims < 2) {
    std::cerr << "ERROR: Output tensor does not have enough dimensions"
              << std::endl;
    ThrowRuntimeException(env, "Output tensor does not have enough dimensions");
    return nullptr;
  }

  int outputHeight = TfLiteTensorDim(outputTensor, 1);
  int outputWidth = TfLiteTensorDim(outputTensor, 2);
  int outputChannels = TfLiteTensorDim(outputTensor, 3);
  if (outputHeight <= 0 || outputWidth <= 0 || outputChannels <= 0) {
    std::cerr << "ERROR: Invalid output tensor dimensions" << std::endl;
    ThrowRuntimeException(env, "Invalid output tensor dimensions");
    return nullptr;
  }

  // Get output tensor data
  float *outputData = static_cast<float *>(TfLiteTensorData(outputTensor));
  if (!outputData) {
    std::cerr << "ERROR: Failed to get output tensor data pointer" << std::endl;
    ThrowRuntimeException(env, "Failed to get output tensor data pointer");
    return nullptr;
  }

  std::vector<detect_result_t> results;

  int numClasses = outputHeight - 4; // Total features minus 4 box coordinates
  int numAnchors = outputWidth;      // Number of anchor points

  // Scale factors to convert from input tensor size back to original image
  // dimensions
  float scaleX =
      static_cast<float>(input_img->cols) / static_cast<float>(inputWidth);
  float scaleY =
      static_cast<float>(input_img->rows) / static_cast<float>(inputHeight);

  for (int anchor = 0; anchor < numAnchors; anchor++) {
    // Extract box coordinates (x_center, y_center, width, height)
    float x_center = outputData[anchor + 0 * numAnchors];
    float y_center = outputData[anchor + 1 * numAnchors];
    float width = outputData[anchor + 2 * numAnchors];
    float height = outputData[anchor + 3 * numAnchors];

    // Find the class with highest confidence
    float maxConfidence = 0.0f;
    int bestClass = -1;

    for (int cls = 0; cls < numClasses; cls++) {
      float confidence = outputData[anchor + (4 + cls) * numAnchors];
      if (confidence > maxConfidence) {
        maxConfidence = confidence;
        bestClass = cls;
      }
    }

    // Skip if no class has confidence above threshold
    if (maxConfidence < boxThresh) {
      continue;
    }

    // Convert center coordinates and dimensions to corner coordinates
    float x1 = x_center - width / 2.0f;
    float y1 = y_center - height / 2.0f;
    float x2 = x_center + width / 2.0f;
    float y2 = y_center + height / 2.0f;

    // Convert to detect_result_t format
    detect_result_t detection;

    // Scale coordinates back to original image size
    detection.box.left = static_cast<int>(x1 * scaleX);
    detection.box.top = static_cast<int>(y1 * scaleY);
    detection.box.right = static_cast<int>(x2 * scaleX);
    detection.box.bottom = static_cast<int>(y2 * scaleY);

    // Clamp coordinates to image boundaries
    detection.box.left =
        std::max(0, std::min(detection.box.left, input_img->cols - 1));
    detection.box.top =
        std::max(0, std::min(detection.box.top, input_img->rows - 1));
    detection.box.right =
        std::max(0, std::min(detection.box.right, input_img->cols - 1));
    detection.box.bottom =
        std::max(0, std::min(detection.box.bottom, input_img->rows - 1));

    // Ensure valid bounding box
    if (detection.box.right <= detection.box.left ||
        detection.box.bottom <= detection.box.top) {
      continue;
    }

    detection.obj_conf = maxConfidence;
    detection.id = bestClass;

    results.push_back(detection);
  }

  // Convert results to Java object array
  jobjectArray resultArray =
      env->NewObjectArray(results.size(), detectionResultClass, nullptr);
  if (!resultArray) {
    std::cerr << "ERROR: Failed to create result array" << std::endl;
    ThrowRuntimeException(env, "Failed to create result array");
    return nullptr;
  }

  for (size_t i = 0; i < results.size(); i++) {
    jobject jResult = MakeJObject(env, results[i]);
    if (!jResult) {
      std::cerr << "ERROR: Failed to create Java object for result " << i
                << std::endl;
      ThrowRuntimeException(
          env, "Failed to create Java object for detection result");
      continue;
    }
    env->SetObjectArrayElement(resultArray, i, jResult);
    env->DeleteLocalRef(jResult);
  }

  std::printf("INFO: Detection completed, found %zu results\n", results.size());
  return resultArray;
}
} // extern "C"
