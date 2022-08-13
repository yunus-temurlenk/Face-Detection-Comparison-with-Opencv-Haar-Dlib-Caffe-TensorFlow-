#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

int main()
{

    //haar model
    std::string faceCascadePath = "/home/code/Downloads/models/haar/haarcascade_frontalface_default.xml";

    cv::CascadeClassifier faceCascade;
    faceCascade.load( faceCascadePath );


    //caffe model
    const std::string caffeConfigFile = "/home/code/Downloads/models/caffe/deploy.prototxt";
    const std::string caffeWeightFile = "/home/code/Downloads/models/caffe/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    cv::dnn::Net net;

    net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);


    //tensorflow model
    const std::string tensorFlowConfigFile = "/home/code/Downloads/models/tensorflow/opencv_face_detector.pbtxt";
    const std::string tensorFlowWeightFile = "/home/code/Downloads/models/tensorflow/opencv_face_detector_uint8.pb";

    cv::dnn::Net net2;

    net2 = cv::dnn::readNetFromTensorflow(tensorFlowWeightFile,tensorFlowConfigFile);

    net2.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net2.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);


    //hog model

    dlib::frontal_face_detector hogDetector = dlib::get_frontal_face_detector();





    std::vector<cv::Rect> haarFaces;


    cv::Mat img;
    cv::namedWindow("Face Detection Comparison",0);


    cv::VideoCapture cap(-1);

    cv::Mat haarMat,caffeMat,tensorFlowMat,hogMat,tempHog;

    int sumCaffe = 0, sumTensor = 0, sumHog = 0;
    while(1)
    {

        cap>>img;

        if(img.data)
        {
            haarMat = img.clone();
            caffeMat = img.clone();
            tensorFlowMat = img.clone();
            hogMat = img.clone();
            tempHog = img.clone();

            /////haar implementation

            faceCascade.detectMultiScale(img, haarFaces);
            for ( size_t i = 0; i < haarFaces.size(); i++ )
            {

                cv::rectangle(haarMat,haarFaces[i],cv::Scalar(0,255,255),5);
            }

            cv::putText(haarMat,"Haar",cv::Point(50,75),1,3,cv::Scalar(0,255,255),2);


            /////caffe implementation

            cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1, cv::Size(300, 300));

            net.setInput(inputBlob);
            cv::Mat detection = net.forward();

            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

            for(int i = 0; i < detectionMat.rows; i++)
            {
                float confidence = detectionMat.at<float>(i, 2);

                if(confidence > 0.7)
                {
                    sumCaffe++;
                    int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
                    int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
                    int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
                    int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

                    cv::rectangle(caffeMat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2);
                    break;
                }
            }
            cv::putText(caffeMat,"Caffe:  " +std::to_string(sumCaffe),cv::Point(50,75),1,3,cv::Scalar(0,255,255),2);




            /////tensorflow implementation

            cv::Mat inputBlob2 = cv::dnn::blobFromImage(img, 1, cv::Size(300, 300));

            net2.setInput(inputBlob2);
            cv::Mat detection2 = net2.forward();

            cv::Mat detectionMat2(detection2.size[2], detection2.size[3], CV_32F, detection2.ptr<float>());



            for(int i = 0; i < detectionMat2.rows; i++)
            {
                float confidence = detectionMat2.at<float>(i, 2);

                if(confidence > 0.7)
                {
                    sumTensor++;
                    int x1 = static_cast<int>(detectionMat2.at<float>(i, 3) * img.cols);
                    int y1 = static_cast<int>(detectionMat2.at<float>(i, 4) * img.rows);
                    int x2 = static_cast<int>(detectionMat2.at<float>(i, 5) * img.cols);
                    int y2 = static_cast<int>(detectionMat2.at<float>(i, 6) * img.rows);

                    cv::rectangle(tensorFlowMat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2);
                    break;
                }
            }
            cv::putText(tensorFlowMat,"tensorFlow:  " +std::to_string(sumTensor),cv::Point(50,75),1,3,cv::Scalar(0,255,255),2);




            int inWidth = (int)((img.cols / (float)img.rows) * 200);

            float scaleHeight = img.rows / (float)200;
            float scaleWidth = img.cols / (float)inWidth;

            resize(hogMat, tempHog, cv::Size(inWidth, 200));

            dlib::cv_image<dlib::bgr_pixel> dlibIm(tempHog);

            std::vector<dlib::rectangle> faceRects = hogDetector(dlibIm);

            for ( size_t i = 0; i < faceRects.size(); i++ )
            {
                sumHog++;
                int x1 = (int)(faceRects[i].left() * scaleWidth);
                int y1 = (int)(faceRects[i].top() * scaleHeight);
                int x2 = (int)(faceRects[i].right() * scaleWidth);
                int y2 = (int)(faceRects[i].bottom() * scaleHeight);
                cv::rectangle(hogMat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,0), (int)(img.cols/150.0), 4);
                break;
            }
            cv::putText(hogMat,"Dlib:  " +std::to_string(sumHog),cv::Point(50,75),1,3,cv::Scalar(0,255,255),2);


            cv::Mat concat1;
            hconcat(haarMat, caffeMat, concat1);
            cv::Mat concat2;
            hconcat(hogMat, tensorFlowMat, concat2);
            cv::Mat concat3;
            vconcat(concat1, concat2, concat3);

            cv::imshow("Face Detection Comparison",concat3);

            int c = cv::waitKey(0);
            if((char)c == 'q')
                break;

        }




    }
    cv::waitKey(0);

    return 0;
}
