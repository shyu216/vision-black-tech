#include <QCoreApplication>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    rs2::context ctx;
    auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
    if (list.size() == 0) 
        throw std::runtime_error("No device detected. Is it plugged in?");
    rs2::device dev = list.front();
    std::cout << "Using device: " << dev.get_info(RS2_CAMERA_INFO_NAME) << ' ' << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Start streaming with default recommended configuration
    pipe.start();

    while (true)
    {
        // Wait for the next set of frames from the camera
        rs2::frameset frames = pipe.wait_for_frames();

        // Get a frame from the depth stream
        rs2::depth_frame depth = frames.get_depth_frame().as<rs2::depth_frame>();

        // Create OpenCV matrix of size (w,h) from the depth data
        cv::Mat depth_image(cv::Size(depth.get_width(), depth.get_height()), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        // Normalize the depth image to fall between 0 and 255 for display purposes
        cv::Mat depth_image_8u;
        depth_image.convertTo(depth_image_8u, CV_8UC1, 255.0 / 10000); // assuming depth is within 10 meters

        // Display the depth image
        cv::imshow("Depth Image", depth_image_8u);

        // Break the loop if the user presses the 'q' key
        if (cv::waitKey(1) == 'q')
        {
            std::cout << "Quitting!" << std::endl;
            break;
        }
    }

    // Stop the RealSense pipeline
    pipe.stop();

    return 0;
}