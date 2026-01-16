#!/usr/bin/env python3
"""
Gear360 Viewer - Simple Python application to display live video stream
"""

import cv2
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Gear360 Video Stream Viewer')
    parser.add_argument('url', nargs='?', 
                       default='http://192.168.43.1:7679/livestream_high.avi',
                       help='Stream URL (default: http://192.168.43.1:7679/livestream_high.avi)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for display (default: 30)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for display window (default: 1.0)')
    
    args = parser.parse_args()
    
    print(f"Connecting to: {args.url}")
    print("Press 'q' or ESC to quit")
    
    # Open video stream - use FFMPEG backend for MJPEG streams
    # The stream is MJPEG format, not standard video container
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    
    # Set low-latency options (similar to FFmpeg flags)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
    
    # Try to open with explicit format hint for MJPEG
    if not cap.isOpened():
        # Retry with format specification
        print("Retrying with MJPEG format hint...")
        # OpenCV doesn't support format hints directly, but we can try
        # using the backend parameter
        cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Error: Could not open stream from {args.url}")
        print("\nTroubleshooting:")
        print("1. Make sure the Gear360 camera is connected and streaming")
        print("2. Check that the URL is correct (try port 7679)")
        print("3. Verify network connectivity")
        print("4. Try: ffplay -i \"" + args.url + "\" to test the stream")
        print("\nNote: This is an MJPEG stream. If OpenCV fails, use ffplay:")
        print(f"  ffplay -hide_banner -fflags nobuffer -flags low_delay -framedrop -i \"{args.url}\"")
        return 1
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Stream properties: {width}x{height} @ {fps:.2f} FPS")
    
    # Calculate display window size
    display_width = int(width * args.scale)
    display_height = int(height * args.scale)
    
    window_name = "Gear360 Viewer - Press 'q' or ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    last_fps_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                print("Stream may have ended or connection lost")
                break
            
            frame_count += 1
            
            # Calculate and display FPS every second
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps_actual = (frame_count - last_fps_count) / (current_time - last_fps_time)
                print(f"FPS: {fps_actual:.2f} | Frames: {frame_count}")
                last_fps_time = current_time
                last_fps_count = frame_count
            
            # Resize frame if scale is not 1.0
            if args.scale != 1.0:
                frame = cv2.resize(frame, (display_width, display_height))
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Check for exit keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nExiting...")
                break
            
            # Control frame rate if needed
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        elapsed = time.time() - start_time
        if elapsed > 0:
            avg_fps = frame_count / elapsed
            print(f"\nStatistics:")
            print(f"  Total frames: {frame_count}")
            print(f"  Total time: {elapsed:.2f} seconds")
            print(f"  Average FPS: {avg_fps:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
