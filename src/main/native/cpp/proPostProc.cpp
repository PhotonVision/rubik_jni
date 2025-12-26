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

/*1: Batch size (a single image was processed).
21000: The total number of potential detection boxes or predictions across all
grid cells/scales (this number can vary based on the input image size and model
architecture). 9: The data points for each prediction, typically representing
the bounding box coordinates, confidence, and class probabilities. The last
dimension of 9 generally corresponds to the following data fields per potential
detection: Columns 1-4: Bounding Box Coordinates x_center: The normalized
X-coordinate of the box center. y_center: The normalized Y-coordinate of the box
center. width: The normalized width of the box. height: The normalized height of
the box. These values are normalized (range from 0 to 1) by the image
dimensions. Column 5: Objectness/Confidence Score A single value representing
the confidence that an object exists within the bounding box. Columns 6-9: Class
Probabilities The probabilities (or scores) for each of the detected classes. In
this case, there are 4 classes (9 total columns - 5 box/confidence columns = 4
class columns).
*/
#include "proPostProc.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <jni.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/version.h>

#include "utils.hpp"

std::vector<DetectResult> proPostProc(TfLiteInterpreter* interpreter,
                                      double boxThresh, double nmsThreshold,
                                      int input_img_width,
                                      int input_img_height) {
  const TfLiteTensor* outputTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);

  const TfLiteQuantizationParams outputParams =
      TfLiteTensorQuantizationParams(outputTensor);

  const int numBoxes = TfLiteTensorDim(outputTensor, 1);
  DEBUG_PRINT("INFO: Detected %d boxes\n", numBoxes);

  // Debug tensor shapes
  DEBUG_PRINT("DEBUG: Output tensor dimensions: ");
#ifndef NDEBUG
  for (int i = 0; i < TfLiteTensorNumDims(outputTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(outputTensor, i));
  }
  std::printf("\n");
#endif

  if (TfLiteTensorType(outputTensor) != kTfLiteInt8) {
    throw std::runtime_error("Expected int8 tensor type");
  }

  int8_t* outputData = static_cast<int8_t*>(TfLiteTensorData(outputTensor));

  const int numPoints = TfLiteTensorDim(outputTensor, 2);
  DEBUG_PRINT("DEBUG: Number of points per box: %d\n", numPoints);

  DEBUG_PRINT("DEBUG: Quantization params - output: zp=%d, scale=%f\n",
              outputParams.zero_point, outputParams.scale);

  std::vector<DetectResult> candidateResults;

  DEBUG_PRINT("DEBUG: Image dimensions: %dx%d\n", input_img_width,
              input_img_height);
  int checked = 0;
  for (int i = 0; i < numBoxes; ++i) {
    int classId = -1;
    float score = -1.0f;
    // Find the class with the highest score

    for (int j = 4; j < numPoints; ++j) {
      int8_t raw_class_score = outputData[j + (numPoints * i)];

      float classScore =
          get_dequant_value(&raw_class_score, kTfLiteInt8, 0,
                            outputParams.zero_point, outputParams.scale);
      if (classScore > score) {
        score = classScore;
        classId = j - 4;
      }

      if (checked <= 5) {
        DEBUG_PRINT(
            "DEBUG: Box %d - class %d score: %.3f classScore %.3f raw %d\n", i,
            classId, score, classScore, raw_class_score);
      }
    }

    checked++;

    if (score < boxThresh) {
      continue;
    }

    // The tensor shape changes, that's fun! We want to calculate this
    // dynamically.
    int8_t raw_x_1_i8 = outputData[i * numPoints + 0];
    int8_t raw_y_1_i8 = outputData[i * numPoints + 1];
    int8_t raw_x_2_i8 = outputData[i * numPoints + 2];
    int8_t raw_y_2_i8 = outputData[i * numPoints + 3];

    // Use proper dequantization for bbox coordinates (like we do for scores)
    float x1 = get_dequant_value(&raw_x_1_i8, kTfLiteInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float y1 = get_dequant_value(&raw_y_1_i8, kTfLiteInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float x2 = get_dequant_value(&raw_x_2_i8, kTfLiteInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float y2 = get_dequant_value(&raw_y_2_i8, kTfLiteInt8, 0,
                                 outputParams.zero_point, outputParams.scale);

    float normal_x1 = x1 * input_img_width;
    float normal_y1 = y1 * input_img_height;
    float normal_x2 = x2 * input_img_width;
    float normal_y2 = y2 * input_img_height;

    float clamped_x1 = std::max(
        0.0f, std::min(normal_x1, static_cast<float>(input_img_width)));
    float clamped_y1 = std::max(
        0.0f, std::min(normal_y1, static_cast<float>(input_img_height)));
    float clamped_x2 = std::max(
        0.0f, std::min(normal_x2, static_cast<float>(input_img_width)));
    float clamped_y2 = std::max(
        0.0f, std::min(normal_y2, static_cast<float>(input_img_height)));
    // Skip bad boxes
    if (clamped_x1 >= clamped_x2 || clamped_y1 >= clamped_y2) {
      continue;
    }

#ifndef NDEBUG
    if (candidateResults.size() < 5) {
      std::printf(" DEBUG: box %d - int8 corners: (%d, %d) to (%d, %d)\n", i,
                  raw_x_1_i8, raw_y_1_i8, raw_x_2_i8, raw_y_2_i8);
      std::printf(
          "DEBUG: box %d - dequantized corners: (%.2f, %.2f) to (%.2f, "
          "%.2f)\n",
          i, x1, y1, x2, y2);
      std::printf(
          "DEBUG: box %d - clamped corners: (%.2f, %.2f) to (%.2f, %.2f), "
          "score=%.3f, class=%d\n",
          i, clamped_x1, clamped_y1, clamped_x2, clamped_y2, score, classId);
    }
#endif

    DetectResult det;
    det.box.x1 = static_cast<int>(std::round(clamped_x1));
    det.box.y1 = static_cast<int>(std::round(clamped_y1));
    det.box.x2 = static_cast<int>(std::round(clamped_x2));
    det.box.y2 = static_cast<int>(std::round(clamped_y2));
    det.box.angle = 0.0;
    det.obj_conf = score;
    det.id = classId;

    candidateResults.push_back(det);
  }

  // NMS
  return optimizedNMS(candidateResults, static_cast<float>(nmsThreshold));
}
