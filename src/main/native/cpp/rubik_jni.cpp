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

typedef struct RubikDetector {
  TfLiteInterpreter *interpreter;
  TfLiteDelegate *delegate;
  TfLiteModel *model;
} RubikDetector;

typedef struct __detect_result_t {
  int id;
  BOX_RECT box;
  float obj_conf;
} detect_result_t;

// Helper function for proper dequantization like example.c
static inline float get_dequant_value(void *data, TfLiteType tensor_type,
                                      int idx, float zero_point, float scale) {
  switch (tensor_type) {
  case kTfLiteUInt8:
    return (static_cast<uint8_t *>(data)[idx] - zero_point) * scale;
  case kTfLiteFloat32:
    return static_cast<float *>(data)[idx];
  default:
    break;
  }
  return 0.0f;
}

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
std::printf("ERROR: detectionResultClass is null!\n");
    return nullptr;
  }

  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");
  if (!constructor) {
    std::printf("ERROR: Could not find constructor for RubikResult!\n");
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
    ThrowRuntimeException(env, "Failed to retrieve model path");
    return 0;
  }

  // Load the model
  TfLiteModel *model = TfLiteModelCreateFromFile(model_name);
  if (!model) {
    ThrowRuntimeException(env, "Failed to load model file");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  std::printf("INFO: Loaded model file '%s'\n", model_name);

  // Create external delegate options
  // We just have to trust that this creates okay, but conveniently if it fails
  // the check when we insert options will catch it.
  TfLiteExternalDelegateOptions delegateOptsValue =
      TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");

  TfLiteExternalDelegateOptions *delegateOpts = &delegateOptsValue;

  // See
  // https://docs.qualcomm.com/bundle/publicresource/topics/80-70014-54/external-delegate-options-for-qnn-delegate.html
  // for what the various delegate options are
  if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "backend_type",
                                          "htp") != kTfLiteOk) {
    ThrowRuntimeException(env, "Failed to set backend type to htp");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "htp_use_conv_hmx",
                                          "1") != kTfLiteOk) {
    ThrowRuntimeException(env, "Failed to enable convolutions");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "htp_performance_mode",
                                          "2") != kTfLiteOk) {
    ThrowRuntimeException(env, "Failed to set htp performance mode");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  // Create the delegate
  TfLiteDelegate *delegate = TfLiteExternalDelegateCreate(delegateOpts);

  if (!delegate) {
    ThrowRuntimeException(env, "Failed to create external delegate");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  } else {
    std::printf("INFO: Created external delegate\n");
  }

  std::printf("INFO: Loaded external delegate\n");

  // Create interpreter options
  TfLiteInterpreterOptions *interpreterOpts = TfLiteInterpreterOptionsCreate();
  if (!interpreterOpts) {
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
    ThrowRuntimeException(env, "Failed to create interpreter");
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  // Modify graph with delegate
  if (TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate) !=
      kTfLiteOk) {
    ThrowRuntimeException(env, "Failed to modify graph with delegate");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  } else {
    std::printf("INFO: Modified graph with external delegate\n");
  }

  // Allocate tensors
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    ThrowRuntimeException(env, "Failed to allocate tensors");
    TfLiteInterpreterDelete(interpreter);
    TfLiteExternalDelegateDelete(delegate);
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  env->ReleaseStringUTFChars(modelPath, model_name);

  // Create RubikDetector object
  RubikDetector *detector = new RubikDetector;
  detector->interpreter = interpreter;
  detector->delegate = delegate;
  detector->model = model;

  // Convert RubikDetector pointer to jlong
  jlong ptr = reinterpret_cast<jlong>(detector);

  std::printf("INFO: TensorFlow Lite initialization completed successfully\n");

  return ptr;
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_rubik_RubikJNI_destroy
  (JNIEnv *env, jclass, jlong ptr)
{
  RubikDetector *detector = reinterpret_cast<RubikDetector *>(ptr);

  if (!detector) {
    ThrowRuntimeException(env, "Invalid RubikDetector pointer");
    return;
  }

  // Now safely use the pointers
  if (detector->interpreter)
    TfLiteInterpreterDelete(detector->interpreter);
  if (detector->delegate)
    TfLiteExternalDelegateDelete(detector->delegate);
  if (detector->model)
    TfLiteModelDelete(detector->model);

  // Delete the RubikDetector object
  delete detector;

  std::printf("INFO: Object Detection instance destroyed successfully\n");
}

inline float calculateIoU(const BOX_RECT &box1, const BOX_RECT &box2) {
  // Calculate intersection coordinates
  const int x1 = std::max(box1.left, box2.left);
  const int y1 = std::max(box1.top, box2.top);
  const int x2 = std::min(box1.right, box2.right);
  const int y2 = std::min(box1.bottom, box2.bottom);

  // No intersection case
  if (x2 <= x1 || y2 <= y1)
    return 0.0f;

  // Calculate areas using pre-computed values when possible
  const int intersectionArea = (x2 - x1) * (y2 - y1);
  const int area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
  const int area2 = (box2.right - box2.left) * (box2.bottom - box2.top);

  return static_cast<float>(intersectionArea) /
         (area1 + area2 - intersectionArea);
}

std::vector<detect_result_t>
optimizedNMS(std::vector<detect_result_t> &candidates, float nmsThreshold) {
  if (candidates.empty())
    return {};

  // Sort by confidence (descending) - single pass
  std::sort(candidates.begin(), candidates.end(),
            [](const detect_result_t &a, const detect_result_t &b) {
              return a.obj_conf > b.obj_conf;
            });

  std::vector<detect_result_t> results;
  results.reserve(candidates.size() / 4); // Reasonable initial capacity

  // Use bitset for faster suppression tracking
  std::vector<bool> suppressed(candidates.size(), false);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i])
      continue;

    // Keep this detection
    results.push_back(candidates[i]);
    const auto &currentBox = candidates[i];

    // Suppress overlapping boxes of the SAME class only
    // Start from i+1 since array is sorted by confidence
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j] || candidates[j].id != currentBox.id)
        continue;

      if (calculateIoU(currentBox.box, candidates[j].box) > nmsThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return results;
}

/*
 * Class:     org_photonvision_rubik_RubikJNI
 * Method:    detect
 * Signature: (JJDD)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_rubik_RubikJNI_detect
  (JNIEnv *env, jobject obj, jlong ptr, jlong input_cvmat_ptr,
   jdouble boxThresh, jdouble nmsThreshold)
{
  RubikDetector *detector = reinterpret_cast<RubikDetector *>(ptr);

  if (!detector) {
    ThrowRuntimeException(env, "Invalid RubikDetector pointer");
    return nullptr;
  }

  if (!detector->interpreter) {
    ThrowRuntimeException(env, "Interpreter not initialized");
    return nullptr;
  }

  TfLiteInterpreter *interpreter = detector->interpreter;
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
  if (input_img->channels() == 3) {
    cv::cvtColor(*input_img, rgb, cv::COLOR_BGR2RGB);
  } else {
    ThrowRuntimeException(env, "Input image must be RGB");
    return nullptr;
  }

  std::memcpy(TfLiteTensorData(input), rgb.data, TfLiteTensorByteSize(input));

  // Start timer for benchmark
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start); // Start timing

  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    ThrowRuntimeException(env, "Interpreter invocation failed");
    return nullptr;
  }

  clock_gettime(CLOCK_MONOTONIC, &end); // End timing

  // Calculate elapsed time in milliseconds
  double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;

  std::printf("INFO: Model execution time: %.2f ms\n", elapsed_time);

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

  // Debug tensor shapes
  std::printf("DEBUG: Boxes tensor dimensions: ");
  for (int i = 0; i < TfLiteTensorNumDims(boxesTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(boxesTensor, i));
  }
  std::printf("\n");

  std::printf("DEBUG: Scores tensor dimensions: ");
  for (int i = 0; i < TfLiteTensorNumDims(scoresTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(scoresTensor, i));
  }
  std::printf("\n");

  std::printf("DEBUG: Classes tensor dimensions: ");
  for (int i = 0; i < TfLiteTensorNumDims(classesTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(classesTensor, i));
  }
  std::printf("\n");

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

  std::printf("DEBUG: Quantization params - boxes: zp=%d, scale=%f\n",
              boxesParams.zero_point, boxesParams.scale);
  std::printf("DEBUG: Quantization params - scores: zp=%d, scale=%f\n",
              scoresParams.zero_point, scoresParams.scale);

  std::vector<detect_result_t> candidateResults;

  std::printf("DEBUG: Image dimensions: %dx%d\n", input_img->cols,
              input_img->rows);

  for (int i = 0; i < numBoxes; ++i) {
    // Use proper dequantization for score
    float score =
        get_dequant_value(scoresData, kTfLiteUInt8, i, scoresParams.zero_point,
                          scoresParams.scale);
    if (score < boxThresh)
      continue;

    int classId = classesData[i];

    // For tensor shape [1, 8400, 4], use sequential indexing per detection
    uint8_t raw_x_center_u8 = boxesData[i * 4 + 0];
    uint8_t raw_y_center_u8 = boxesData[i * 4 + 1];
    uint8_t raw_width_u8 = boxesData[i * 4 + 2];
    uint8_t raw_height_u8 = boxesData[i * 4 + 3];

    // Use proper dequantization for bbox coordinates (like we do for scores)
    float x_center =
        get_dequant_value(&raw_x_center_u8, kTfLiteUInt8, 0,
                          boxesParams.zero_point, boxesParams.scale);
    float y_center =
        get_dequant_value(&raw_y_center_u8, kTfLiteUInt8, 0,
                          boxesParams.zero_point, boxesParams.scale);
    float width = get_dequant_value(&raw_width_u8, kTfLiteUInt8, 0,
                                    boxesParams.zero_point, boxesParams.scale);
    float height = get_dequant_value(&raw_height_u8, kTfLiteUInt8, 0,
                                     boxesParams.zero_point, boxesParams.scale);

    // Calculate corners
    float x1 = x_center;
    float y1 = y_center;
    float x2 = x_center + (width / 2.0f);
    float y2 = y_center + (height / 2.0f);

    float clamped_x1 =
        std::max(0.0f, std::min(x1, static_cast<float>(input_img->cols)));
    float clamped_y1 =
        std::max(0.0f, std::min(y1, static_cast<float>(input_img->rows)));
    float clamped_x2 =
        std::max(0.0f, std::min(x2, static_cast<float>(input_img->cols)));
    float clamped_y2 =
        std::max(0.0f, std::min(y2, static_cast<float>(input_img->rows)));

    // Skip bad boxes
    if (clamped_x1 >= clamped_x2 || clamped_y1 >= clamped_y2) {
      continue;
    }

    if (candidateResults.size() < 5) {
      std::printf(" DEBUG: box %d - uint8: center(%d, %d) size(%d, %d)\n", i,
                  raw_x_center_u8, raw_y_center_u8, raw_width_u8,
                  raw_height_u8);
      std::printf(
          "DEBUG: box %d - dequantized: center(%.2f, %.2f) size(%.2f, %.2f)\n",
          i, x_center, y_center, width, height);
      std::printf("DEBUG: box %d - corners: (%.2f, %.2f) to (%.2f, %.2f)\n", i,
                  x1, y1, x2, y2);
      std::printf(
          "DEBUG: box %d - clamped corners: (%.2f, %.2f) to (%.2f, %.2f), "
          "score=%.3f, class=%d\n",
          i, clamped_x1, clamped_y1, clamped_x2, clamped_y2, score, classId);
    }

    detect_result_t det;
    det.box.left = static_cast<int>(std::round(clamped_x1));
    det.box.top = static_cast<int>(std::round(clamped_y1));
    det.box.right = static_cast<int>(std::round(clamped_x2));
    det.box.bottom = static_cast<int>(std::round(clamped_y2));
    det.obj_conf = score;
    det.id = classId;

    candidateResults.push_back(det);
  }

  // NMS
  std::vector<detect_result_t> results =
      optimizedNMS(candidateResults, static_cast<float>(nmsThreshold));

  jobjectArray jResults =
      env->NewObjectArray(results.size(), detectionResultClass, nullptr);
  for (size_t i = 0; i < results.size(); ++i) {
    jobject jDet = MakeJObject(env, results[i]);
    env->SetObjectArrayElement(jResults, i, jDet);
  }

  return jResults;
}

} // extern "C"
