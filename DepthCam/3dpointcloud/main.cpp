#include <QCoreApplication>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

// Function to convert point cloud to OpenCV image
cv::Mat pointcloud_to_image(const rs2::points &points)
{
    // Get the vertices of the point cloud
    auto vertices = points.get_vertices();
    int width = 640;  // Width of the output image
    int height = 480; // Height of the output image

    // Create an empty image with 3 channels (RGB)
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    for (int i = 0; i < points.size(); ++i)
    {
        // Get the coordinates of the point
        float x = vertices[i].x;
        float y = vertices[i].y;
        float z = vertices[i].z;

        // Map the 3D coordinates to 2D image coordinates (side view)
        int v = static_cast<int>((y + 1.0f) * width / 2.0f);
        int u = static_cast<int>(z * height);

        // Ensure the coordinates are within the image bounds
        if (u >= 0 && u < height && v >= 0 && v < width)
        {
            // Set the pixel value based on the x-coordinate
            float x = vertices[i].x;
            image.at<cv::Vec3b>(v, u) = cv::Vec3b(255 - static_cast<int>(x * 255), 255 - static_cast<int>(x * 255), 255 - static_cast<int>(x * 255));
        }
    }

    return image;
}

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

        // Get depth frame
        rs2::depth_frame depth = frames.get_depth_frame();

        // Generate the point cloud
        rs2::pointcloud pc;
        rs2::points points = pc.calculate(depth);

        // Convert point cloud to OpenCV image
        cv::Mat image = pointcloud_to_image(points);

        // Display the image
        cv::imshow("Point Cloud", image);

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