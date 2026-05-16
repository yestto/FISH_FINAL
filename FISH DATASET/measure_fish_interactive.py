"""
Interactive Fish Measurement Tool
Opens each fish image in a window. You click TWO points on the ruler
to calibrate, then click the fish nose and tail to measure length.

Controls:
  - Click 2 points on the ruler → enter cm distance in terminal
  - Click fish nose, then tail → shows length in cm
  - Press 'n' → next fish
  - Press 'r' → reset clicks
  - Press 'q' → quit and save
"""
import cv2
import numpy as np
import pandas as pd
import os
import glob

class FishMeasurer:
    def __init__(self):
        self.points = []
        self.px_per_cm = None
        self.fish_length_cm = None
        self.fish_height_cm = None
        self.mode = 'calibrate'  # 'calibrate', 'length', 'height'
        self.img = None
        self.display = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            # Draw the clicked point
            cv2.circle(self.display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Fish Measurement', self.display)
            
            if self.mode == 'calibrate' and len(self.points) == 2:
                # Calculate pixel distance between the two ruler points
                p1, p2 = self.points
                dist_px = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                cv2.line(self.display, p1, p2, (255, 0, 0), 2)
                cv2.imshow('Fish Measurement', self.display)
                print(f"  Pixel distance: {dist_px:.0f} px")
                print(f"  Enter the cm distance between these two ruler marks: ", end="")
                
            elif self.mode == 'length' and len(self.points) == 2:
                p1, p2 = self.points
                dist_px = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                cv2.line(self.display, p1, p2, (0, 255, 0), 2)
                self.fish_length_cm = dist_px / self.px_per_cm
                text = f"Length: {self.fish_length_cm:.2f} cm"
                cv2.putText(self.display, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow('Fish Measurement', self.display)
                print(f"  FISH LENGTH: {self.fish_length_cm:.2f} cm ({dist_px:.0f} px)")
                print("  Now click belly → dorsal fin for HEIGHT (or press 'n' for next)")
                self.points = []
                self.mode = 'height'
                
            elif self.mode == 'height' and len(self.points) == 2:
                p1, p2 = self.points
                dist_px = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                cv2.line(self.display, p1, p2, (255, 255, 0), 2)
                self.fish_height_cm = dist_px / self.px_per_cm
                text = f"Height: {self.fish_height_cm:.2f} cm"
                cv2.putText(self.display, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imshow('Fish Measurement', self.display)
                print(f"  FISH HEIGHT: {self.fish_height_cm:.2f} cm ({dist_px:.0f} px)")
                print("  Press 'n' for next fish, 'r' to reset")
                self.points = []


def main():
    root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'
    
    fish_dirs = ['fish01','fish2','fish3','fish4','fish5','fish6','fish7',
                 'fish8','fish9','fish10','fish11','fish12','fish13','fish14','fish15']
    
    results = []
    
    for fish_dir in fish_dirs:
        img_folder = os.path.join(root, fish_dir, 'single image')
        if not os.path.isdir(img_folder):
            continue
        imgs = glob.glob(os.path.join(img_folder, '*.jpg'))
        if not imgs:
            continue
        
        img = cv2.imread(imgs[0])
        if img is None:
            continue
        
        # Resize for display if too large
        h, w = img.shape[:2]
        scale = min(1.0, 1200 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        measurer = FishMeasurer()
        measurer.img = img.copy()
        measurer.display = img.copy()
        
        # Add instructions
        cv2.putText(measurer.display, f"{fish_dir} - Click 2 ruler points to calibrate", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.namedWindow('Fish Measurement', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Fish Measurement', measurer.mouse_callback)
        cv2.imshow('Fish Measurement', measurer.display)
        
        print(f"\n{'='*50}")
        print(f"FISH: {fish_dir}")
        print(f"  Step 1: Click TWO points on the ruler (e.g., 0 and 10)")
        
        while True:
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                # Save whatever we have
                if results:
                    df = pd.DataFrame(results)
                    out = os.path.join(root, 'truth_values_manual.csv')
                    df.to_csv(out, index=False)
                    print(f"\nSaved {len(results)} measurements to: {out}")
                return
            
            elif key == ord('n'):
                # Next fish
                if measurer.fish_length_cm:
                    results.append({
                        'FishID': fish_dir,
                        'Length_cm': round(measurer.fish_length_cm, 2),
                        'Height_cm': round(measurer.fish_height_cm, 2) if measurer.fish_height_cm else None,
                        'PxPerCm': round(measurer.px_per_cm, 2) if measurer.px_per_cm else None,
                    })
                    print(f"  ✅ Saved: L={measurer.fish_length_cm:.2f}cm H={measurer.fish_height_cm}")
                break
            
            elif key == ord('r'):
                # Reset
                measurer.points = []
                measurer.mode = 'calibrate'
                measurer.px_per_cm = None
                measurer.fish_length_cm = None
                measurer.fish_height_cm = None
                measurer.display = img.copy()
                cv2.putText(measurer.display, f"{fish_dir} - Click 2 ruler points to calibrate",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow('Fish Measurement', measurer.display)
                print("  RESET - Click 2 ruler points again")
            
            elif key == 13:  # Enter key - for calibration input
                if measurer.mode == 'calibrate' and len(measurer.points) == 2:
                    try:
                        cm_dist = float(input())
                        p1, p2 = measurer.points
                        px_dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                        measurer.px_per_cm = px_dist / cm_dist
                        print(f"  ✅ Calibrated: PxPerCm = {measurer.px_per_cm:.2f}")
                        print(f"  Step 2: Now click fish NOSE → TAIL for length")
                        measurer.points = []
                        measurer.mode = 'length'
                        
                        # Update display
                        text = f"PxPerCm={measurer.px_per_cm:.1f} - Click nose then tail"
                        cv2.putText(measurer.display, text, (10, 55), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        cv2.imshow('Fish Measurement', measurer.display)
                    except:
                        print("  Invalid input. Enter a number (e.g., 10)")
    
    cv2.destroyAllWindows()
    
    if results:
        df = pd.DataFrame(results)
        out = os.path.join(root, 'truth_values_manual.csv')
        df.to_csv(out, index=False)
        print(f"\n✅ Saved {len(results)} measurements to: {out}")
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
