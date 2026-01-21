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

#pragma once

#include <vector>

#include <tensorflow/lite/c/c_api.h>

/**
 * Enumeration of model versions. This matches the enum in the Java code, and
 * when one is updated the other should be as well. Note that YOLOV5 is omitted
 * since it is not supported.
 */
enum ModelVersion { YOLOV8 = 1, YOLOV11 = 2 };

struct BoxRect {
  int x1;
  int y1;
  int x2;
  int y2;
  double angle;
};

struct RubikDetector {
  TfLiteInterpreter* interpreter;
  TfLiteDelegate* delegate;
  TfLiteModel* model;
  ModelVersion version;
};

struct DetectResult {
  int id;
  BoxRect box;
  float obj_conf;
};

#ifndef NDEBUG
#define DEBUG_PRINT(...) std::printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) \
  do {                   \
  } while (0)
#endif

/**
 * Gets the dequantized value from tensor data.
 * @param data Pointer to the tensor data.
 * @param tensor_type The type of the tensor.
 * @param idx The index of the value to retrieve.
 * @param zero_point The zero point for dequantization.
 * @param scale The scale for dequantization.
 * @return The dequantized float value.
 */
float get_dequant_value(void* data, TfLiteType tensor_type, int idx,
                        float zero_point, float scale);

/**
 * Infers the width, height, and channels of a tensor as if it were an image.
 * @param tensor The TensorFlow Lite tensor.
 * @param w Pointer to store the width.
 * @param h Pointer to store the height.
 * @param c Pointer to store the number of channels.
 * @return True if the dimensions were successfully inferred, false otherwise.
 */
bool tensor_image_dims(const TfLiteTensor* tensor, int* w, int* h, int* c);

/**
 * Performs Non-Maximum Suppression (NMS) on a list of detection results.
 *
 * @param candidates The list of detection candidates.
 * @param nmsThreshold The IoU threshold for suppression.
 * @return A vector of filtered detection results.
 */
std::vector<DetectResult> optimizedNMS(std::vector<DetectResult>& candidates,
                                       float nmsThreshold);
