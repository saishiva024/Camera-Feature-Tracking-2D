/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    /* ****************** DISABLE THIS IF VISUALIZATION IS NOT REQUIRED ****************** */
    bool bVis = true;            // visualize results
    /* *********************************************************************************** */

    vector<AlgoCharacteristics> algoCharacteristics;
    InitAlgoCombinations(algoCharacteristics);

    for(auto &algoCharacteristic : algoCharacteristics)
    {
        dataBuffer.clear();

        /* MAIN LOOP OVER ALL IMAGES */

        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
        {
            /* LOAD IMAGE INTO BUFFER */

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file and convert to grayscale
            cv::Mat img, imgGray;
            img = cv::imread(imgFullFilename);
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

            //// STUDENT ASSIGNMENT
            //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = imgGray;

            if(dataBuffer.size() == dataBufferSize)
            {
                dataBuffer.erase(dataBuffer.begin());
            }

            dataBuffer.push_back(frame);

            //// EOF STUDENT ASSIGNMENT
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

            /* DETECT IMAGE KEYPOINTS */

            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image
            //string detectorType = "HARRIS"; //SHITOMASI, HARRIS
            string detectorType = algoCharacteristic.detector;

            //// STUDENT ASSIGNMENT
            //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
            //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

            if (detectorType.compare("SHITOMASI") == 0)
            {
                detKeypointsShiTomasi(keypoints, imgGray, algoCharacteristic.detectorElapsedTime.at(imgIndex), false);
            }
            else if(detectorType.compare("HARRIS") == 0)
            {
                detKeypointsHarris(keypoints, imgGray, algoCharacteristic.detectorElapsedTime.at(imgIndex), false);
            }
            else
            {
                detKeypointsModern(keypoints, imgGray, detectorType, algoCharacteristic.detectorElapsedTime.at(imgIndex), false);
            }

            algoCharacteristic.numKpts.at(imgIndex) = keypoints.size();
            //// EOF STUDENT ASSIGNMENT

            //// STUDENT ASSIGNMENT
            //// TASK MP.3 -> only keep keypoints on the preceding vehicle

            // only keep keypoints on the preceding vehicle
            bool bFocusOnVehicle = true;
            cv::Rect vehicleRect(535, 180, 180, 150);

            vector<cv::KeyPoint> precedingVehicleKeypoints;

            if (bFocusOnVehicle)
            {
                for(auto keypoint : keypoints)
                {
                    cv::Point2f point = cv::Point2f(keypoint.pt);
                    if(vehicleRect.contains(point))
                    {
                        precedingVehicleKeypoints.push_back(keypoint);
                    }
                }

                keypoints = precedingVehicleKeypoints;

                algoCharacteristic.numKptsVehicle.at(imgIndex) = precedingVehicleKeypoints.size();
            }


            //// EOF STUDENT ASSIGNMENT

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;
            cout << "#2 : DETECT KEYPOINTS done" << endl;

            /* EXTRACT KEYPOINT DESCRIPTORS */

            //// STUDENT ASSIGNMENT
            //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
            //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

            cv::Mat descriptors;
            //string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
            string descriptorType = algoCharacteristic.descriptor;

            descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, algoCharacteristic.descriptorElapsedTime.at(imgIndex));

            algoCharacteristic.numDescriptors.at(imgIndex) = descriptors.size().width * descriptors.size().height;
            //// EOF STUDENT ASSIGNMENT

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {

                /* MATCH KEYPOINT DESCRIPTORS */

                vector<cv::DMatch> matches;
                // string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                // string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
                // string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

                string matcherType = algoCharacteristic.matcher;
                string descriptorType = (algoCharacteristic.descriptor.compare("SIFT") == 0) ? "DES_HOG" : "DES_BINARY";
                string selectorType = algoCharacteristic.selector;

                //// STUDENT ASSIGNMENT
                //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                matches, descriptorType, matcherType, selectorType, 
                                algoCharacteristic.matcherElapsedTime.at(imgIndex));
                cout<<"Matched " << matches.size() << "keypoints";

                algoCharacteristic.numMatchedKpts.at(imgIndex) = matches.size();

                //// EOF STUDENT ASSIGNMENT

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                // visualize matches between current and previous image
                //bVis = true;
                if (bVis)
                {
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                    (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                    matches, matchImg,
                                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);
                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0); // wait for key to be pressed
                }
                //bVis = false;
            }

        } // eof loop over all images

    }
    summarizeAndGenerateOutputReports(algoCharacteristics);
    return 0;
}

void InitAlgoCombinations(vector<AlgoCharacteristics>& algoChars)
 {
    vector<string> detectors = { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE",  "SIFT" };
    vector<string> descriptors = { "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };
    vector<string> matchers = { "MAT_BF" };
    vector<string> selectors = { "SEL_KNN" };

    for (auto detector : detectors) 
    {
        for (auto descriptor : descriptors)
        {
            for (auto matcher : matchers) 
            {
                for (auto selector : selectors) 
                {
                    if ((descriptor.compare("AKAZE") == 0) && (detector.compare("AKAZE") != 0) || (descriptor.compare("ORB") == 0) && (detector.compare("SIFT") == 0))
                    { 
                        continue; 
                    }

                    AlgoCharacteristics ac;
                    ac.detector = detector;
                    ac.descriptor = descriptor;
                    ac.matcher = matcher;
                    ac.selector = selector;

                    algoChars.push_back(ac);
                }
            }
        }
    }
}

void summarizeAndGenerateOutputReports(vector<AlgoCharacteristics> &algoOutputs)
 {
    ofstream reportFile{"../report/Report.csv"};
    ofstream summaryFile{"../report/Summary.csv"};

    int totalKptsPerAllImages;
    int totalKptsVehiclePerAllImages;
    int totalDescriptorsPerAllImages;
    int totalMatchedKptsPerAllImages;
    double totalDetectorDescritorTimeElapsed;
    double totalTimeElapsed;
    string algoCombo;

    reportFile << "IMAGE, DETECTOR, DESCRIPTOR, MATCHER, SELECTOR, #KEYPOINTS, #KEYPOINTS ON VEHICLE, #DESCRIPTORS, #MATCHED KEYPOINTS, DETECTOR ELAPSED TIME, DESCRIPTOR ELAPSED TIME, MATCHER ELAPSED TIME" << endl;
    summaryFile << "ALGO COMBO, #KEYPOINTS, #KEYPOINTS ON VEHICLE, #DESCRIPTORS, #MATCHED KEYPOINTS, DETECTOR & DESCRIPTOR TIME(ms), TOTAL TIME ELAPSED(ms)" << endl;

    for (auto &algoOutput : algoOutputs)
    {
        totalKptsPerAllImages = 0;
        totalKptsVehiclePerAllImages = 0;
        totalDescriptorsPerAllImages = 0;
        totalMatchedKptsPerAllImages = 0;
        totalDetectorDescritorTimeElapsed = 0.0;
        totalTimeElapsed = 0.0;
        algoCombo = algoOutput.detector + "+" + algoOutput.descriptor;

        for(int imIndex = 0; imIndex < 10; imIndex++)
        {
            reportFile << imIndex << ", " 
                       << algoOutput.detector << ", " 
                       << algoOutput.descriptor << ", " 
                       << algoOutput.matcher << ", " 
                       << algoOutput.selector << ", " 
                       << algoOutput.numKpts.at(imIndex)<< ", "
                       << algoOutput.numKptsVehicle.at(imIndex) << ", " 
                       << algoOutput.numDescriptors.at(imIndex) << ", " 
                       << algoOutput.numMatchedKpts.at(imIndex) << ", " 
                       << algoOutput.detectorElapsedTime.at(imIndex) << ", " 
                       << algoOutput.descriptorElapsedTime.at(imIndex) << ", " 
                       << algoOutput.matcherElapsedTime.at(imIndex) << endl;

            totalKptsPerAllImages += algoOutput.numKpts.at(imIndex);
            totalKptsVehiclePerAllImages += algoOutput.numKptsVehicle.at(imIndex);
            totalDescriptorsPerAllImages += algoOutput.numDescriptors.at(imIndex);
            totalMatchedKptsPerAllImages += algoOutput.numMatchedKpts.at(imIndex);
            totalDetectorDescritorTimeElapsed += (algoOutput.detectorElapsedTime.at(imIndex) + 
                                                 algoOutput.descriptorElapsedTime.at(imIndex)) ;
            totalTimeElapsed += totalDetectorDescritorTimeElapsed + algoOutput.matcherElapsedTime.at(imIndex);
        }
        summaryFile << algoCombo << ", " 
                    << totalKptsPerAllImages << ", " 
                    << totalKptsVehiclePerAllImages << ", " 
                    << totalDescriptorsPerAllImages << ", " 
                    << totalMatchedKptsPerAllImages << ", " 
                    << totalDetectorDescritorTimeElapsed << ", " 
                    << totalTimeElapsed << endl;
    }

    reportFile.close();
    summaryFile.close();

    std::cout << "Report Generated at " << "../report/Report.csv" << std::endl;
    std::cout << "Summary Generated at " << "../report/Summary.csv" << std::endl;
}