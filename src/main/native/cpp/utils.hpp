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

typedef struct _BOX_RECT {
  int x1;
  int x2;
  int y1;
  int y2;
  double angle;
} BOX_RECT;

typedef struct __detect_result_t {
  int id;
  BOX_RECT box;
  float obj_conf;
} detect_result_t;

/**
 * Performs Non-Maximum Suppression (NMS) on a list of detection results.
 *
 * @param candidates The list of detection candidates.
 * @param nmsThreshold The IoU threshold for suppression.
 * @return A vector of filtered detection results.
 */
std::vector<detect_result_t>
optimizedNMS(std::vector<detect_result_t> &candidates, float nmsThreshold);
