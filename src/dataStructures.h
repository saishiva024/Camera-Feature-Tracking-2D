#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

struct AlgoCharacteristics {
    std::string detector;
    std::string descriptor;
    std::string matcher;
    std::string selector;
    std::array<int, 10> numKpts;
    std::array<int, 10> numKptsVehicle;
    std::array<int, 10> numDescriptors;
    std::array<int, 10> numMatchedKpts;
    std::array<double, 10> detectorElapsedTime;
    std::array<double, 10> descriptorElapsedTime;
    std::array<double, 10> matcherElapsedTime;
};


#endif /* dataStructures_h */
