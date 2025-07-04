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

 import org.opencv.core.Point;
 import org.opencv.core.Rect2d;
 
 public class RubikJNI {

 
     public static class RubikResult {
         public RubikResult(
             int left, int top, int right, int bottom, float conf, int class_id
         ) {
             this.conf = conf;
             this.class_id = class_id;
             this.rect = new Rect2d(new Point(left, top), new Point(right, bottom));
         }
         
         public final Rect2d rect;
         public final float conf;
         public final int class_id;
 
         @Override
         public String toString() {
             return "RubikResult [rect=" + rect + ", conf=" + conf + ", class_id=" + class_id + "]";
         }
 
         @Override
         public int hashCode() {
             final int prime = 31;
             int result = 1;
             result = prime * result + ((rect == null) ? 0 : rect.hashCode());
             result = prime * result + Float.floatToIntBits(conf);
             result = prime * result + class_id;
             return result;
         }
 
         @Override
         public boolean equals(Object obj) {
             if (this == obj)
                 return true;
             if (obj == null)
                 return false;
             if (getClass() != obj.getClass())
                 return false;
             RubikResult other = (RubikResult) obj;
             if (rect == null) {
                 if (other.rect != null)
                     return false;
             } else if (!rect.equals(other.rect))
                 return false;
             if (Float.floatToIntBits(conf) != Float.floatToIntBits(other.conf))
                 return false;
             if (class_id != other.class_id)
                 return false;
             return true;
         }
     }
 
     /**
      * TODO: Determine what needs to go here based on native code
      */
     public static native long create();
     
     /**
      * Delete all native resources assocated with a detector
      TODO: Determine what needs to go here based on native code
      */
     public static native long destroy();
 
     /**
      * Run detction
    TODO: Determine what needs to go here based on native code
      */
     public static native RubikResult[] detect();
 }