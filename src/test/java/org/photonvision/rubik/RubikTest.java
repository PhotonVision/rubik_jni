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

package org.photonvision.rubik;

import java.io.IOException;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.rubik.RubikJNI.RubikResult;
import edu.wpi.first.util.CombinedRuntimeLoader;

public class RubikTest {
    @Test
    public void testBasicBlobs() {
        try {
            CombinedRuntimeLoader.loadLibraries(RubikTest.class, Core.NATIVE_LIBRARY_NAME);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(Core.getBuildInformation());
        System.out.println(Core.OpenCLApiCallError);

        System.out.println("Loading rubik_jni");
        System.load("/home/ubuntu/rubik_jni/cmake_build/librubik_jni.so");

        System.out.println("Loading bus");
        Mat img = Imgcodecs.imread("src/test/resources/bus.jpg");

        if (img.empty()) {
            throw new RuntimeException("Failed to load image");
        }

        System.out.println("Image loaded: " + img.size() + " " + img.type());

        System.out.println("Creating Rubik detector");
        long ptr = RubikJNI.create("src/test/resources/yolov8nCOCOv2.tflite");

            if (ptr == 0) {
                throw new RuntimeException("Failed to create Rubik detector");
            }


        System.out.println("Rubik detector created: " + ptr);
        RubikResult[] ret = RubikJNI.detect(ptr, img.getNativeObjAddr(), 0.5f, 0.45f);

        System.out.println("Detection results: " + Arrays.toString(ret));

        System.out.println("Releasing Rubik detector");
        RubikJNI.destroy(ptr);

        for (RubikResult result : ret) {
            System.out.println("Result: " + result);

            Scalar color = new Scalar(0, 255, 0); // Green color is default for bounding box

            if(result.class_id == 0) {
                color = new Scalar(255, 0, 0); // Blue for person
            } else if (result.class_id == 5) {
                color = new Scalar(0, 0, 255); // Red for bus
            }

            // Draw bounding box on the image
            Imgproc.rectangle(
                img,
                new Point(result.rect.x, result.rect.y),
                new Point(result.rect.x + result.rect.width, result.rect.y + result.rect.height),
                color,
                2 // Thickness
            );

            // Put label text
            // Imgproc.putText(
            //     img,
            //     result.class_id + " " + String.format("%.2f", result.conf),
            //     new Point(result.rect.x, result.rect.y - 10),
            //     Imgproc.FONT_HERSHEY_SIMPLEX,
            //     0.5, // Font scale
            //     new Scalar(0, 255, 0), // Green color
            //     1 // Thickness
            // );
        }

        // Save the image with results
        Imgcodecs.imwrite("src/test/resources/bus_with_results.jpg", img);
        System.out.println("Results written to image and saved as bus_with_results.jpg");
    }
}
