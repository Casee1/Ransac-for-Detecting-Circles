// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <vector>

using namespace std;

wchar_t* projectPath;

bool inside(Mat src, int i, int j)
{
	if (i < 0 || j < 0 || i >= src.rows || j >= src.cols)
	{
		return false;
	}

	return true;
}

vector<int> calcHist(Mat_<uchar> img) {
	vector<int> hist(256);
	int height = img.rows;
	int width = img.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			hist[img(i, j)]++;
		}
	}
	return hist;
}


Mat_<float> conv(Mat_<uchar> src, Mat_<float> H)
{
	int height = src.rows;
	int width = src.cols;
	int w = H.cols;
	float sum = 0.0f;
	Mat_<float> dst = Mat(height, width, CV_32SC1, Scalar(0));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			sum = 0.0f;
			for (int u = 0; u < H.rows; u++)
			{
				for (int v = 0; v < w; v++)
				{
					int i2 = i + u - H.rows / 2;
					int j2 = j + v - w / 2;
					if (inside(src, i2, j2))
					{
						sum += H(u, v) * src(i2, j2);
					}
				}
			}
			dst(i, j) = sum;
		}
	}

	return dst;
}

//lab11
//1. create the Sobel kernels Sx and Sy (float)
//2. calculate the derivative in x and y direction
// dx = src Sx
// dy = src Sy
// - floating point images,without normalization , for visualization imshow("dx", abs(dx)/255)
//3. calculate the magnitude and angle
// mag = sqrt(dx * dx + dy* dy)
// angle - atan2(dy, dx)
//- for visualization imshow("mag", abs(mag)/255)
//4. non-maximum suppression = thinning
//quantize angles q = ((int) round(angle/(2pi)*8))%8
// if(mag(i,j)) > than neighbor its neighbor in direction q and (q+4)%8, mag2(i,j) = mag(i,j)
// otherwise erase (mag2(i,j) = 0)
//5. classify edges = adaptive thresholding
//magn = mag/(4*sqrt(2))
//NoNonEdge = (1-p) * (Height * Width - Hist[0]) where p = 0.01 - 0.1
//t_high = first i for which sum(hist[i]) > NoNonEdge ~59
//t_low = 0.4 * t_high

//everything above t_high = strong edge = 255
// between t_low and t_high = weak edge = 128
// below t_low = non-edge = 0
//6. edge extension
// //for each string edge mark all of its weak edge neighbors as strong edges recursively 

Mat_<uchar> cannyEdgeDetection(Mat src, float p)
{
	Mat_<float> Sx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat_<float> Sy = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

	imshow("src", src);
	int height = src.rows;
	int width = src.cols;
	auto dx = conv(src, Sx);
	auto dy = conv(src, Sy);

	//imshow("dx", abs(dx) / 255);

	Mat_<float> mag(height, width);
	Mat_<float> angle(height, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{

			mag(i, j) = sqrt(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
			angle(i, j) = atan2(dy(i, j), dx(i, j));
			if (angle(i, j) >= 0)
			{
				angle(i, j) += 2 * CV_PI;
			}
		}
	}

	//imshow("mag", abs(mag) / 255);

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Mat_<float> mag2(height, width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int q = ((int)round(angle(i, j) / (2 * CV_PI) * 8)) % 8;
			int i2 = i + di[q];
			int j2 = j + dj[q];
			int i3 = i + di[(q + 4) % 8];
			int j3 = j + dj[(q + 4) % 8];
			if (inside(mag, i2, j2) && inside(mag, i3, j3))
			{

				if (mag(i, j) > mag(i2, j2) && mag(i, j) > mag(i3, j3))
				{
					mag2(i, j) = mag(i, j);
				}
				else
				{
					mag2(i, j) = 0;
				}
			}
		}
	}

	//imshow("mag2", mag2);
	Mat_<float> magn(height, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			magn(i, j) = mag2(i, j) / (4 * sqrt(2)); //thinned version
		}
	}
	//imshow("magn", magn);

	auto H = calcHist(magn);

	float NoNonEdge = (1 - p) * (height * width - H[0]);

	int t_high = 0;
	int sum = 0;
	for (int i = 1; i < 256; i++)
	{
		sum = sum + H[i];
		if (sum > NoNonEdge)
		{
			t_high = i;
			i = 256;
		}
	}

	int t_low = 0.4 * t_high;

	printf("%d\n", t_high);
	printf("%d\n", t_low);

	Mat_<uchar> dst(height, width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (magn(i, j) > t_high)
			{
				dst(i, j) = 255;
			}
			else if (magn(i, j) <= t_high && magn(i, j) >= t_low)
			{
				dst(i, j) = 128;
			}
			else if (magn(i, j) < t_low)
			{
				dst(i, j) = 0;
			}
		}
	}
	std::queue<Point> Q;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst(i, j) == 255)
			{
				Q.push(Point(j, i));
			}
		}
	}

	while (!Q.empty())
	{
		Point p = Q.front();
		Q.pop();
		for (int k = 0; k < 8; k++)
		{
			if (inside(dst, p.y + di[k], p.x + dj[k]) && dst(p.y + di[k], p.x + dj[k]) == 128)
			{
				dst(p.y + di[k], p.x + dj[k]) = 255;
				Q.push(Point(p.x + dj[k], p.y + di[k]));
			}
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst(i, j) == 128)
			{
				dst(i, j) = 0;
			}
		}
	}

	imshow("Canny Edge Detection", dst);
	return dst;
}

void RANSAC_DeterminantCircle()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		float edgeThresh = 0.2f;
		Mat_<uchar> edges = cannyEdgeDetection(src, edgeThresh);

		vector<Point2i> v;

		for (int i = 0; i < edges.rows; i++)
		{
			for (int j = 0; j < edges.cols; j++)
			{
				if (edges(i, j) == 255)
				{
					v.push_back(Point2i(j, i));
				}
			}
		}

		int n = v.size();
		double p = 0.99;
		double q = 0.5;
		int T = q * n;
		int s = 3;
		int N = log(1 - p) / log(1 - pow(q, s));
		double t = 5.0;

		int maxInliers = 0;
		Point2f best_center;
		float best_radius = 0;

		for (int trial = 0; trial < N; trial++)
		{
			int i = rand() % n;
			int j = rand() % n;
			int k = rand() % n;

			while (i == j)
			{
				j = rand() % n;
			}
			while (i == k || j == k)
			{
				k = rand() % n;
			}

			Point2f p1 = v[i];
			Point2f p2 = v[j];
			Point2f p3 = v[k];

			float A = p1.x * (p2.y - p3.y) - p1.y * (p2.x - p3.x) + (p2.x * p3.y - p3.x * p2.y);

			if (abs(A) < 1e-6)
			{
				continue;
			}

			float Bx = (p1.x * p1.x + p1.y * p1.y) * (p3.y - p2.y) +
				(p2.x * p2.x + p2.y * p2.y) * (p1.y - p3.y) +
				(p3.x * p3.x + p3.y * p3.y) * (p2.y - p1.y);

			float By = (p1.x * p1.x + p1.y * p1.y) * (p2.x - p3.x) +
				(p2.x * p2.x + p2.y * p2.y) * (p3.x - p1.x) +
				(p3.x * p3.x + p3.y * p3.y) * (p1.x - p2.x);

			float cx = -Bx / (2 * A);
			float cy = -By / (2 * A);
			float r = sqrt((cx - p1.x) * (cx - p1.x) + (cy - p1.y) * (cy - p1.y));

			int inliers = 0;
			for (auto& pts : v)
			{
				float dist = sqrt((pts.x - cx) * (pts.x - cx) + (pts.y - cy) * (pts.y - cy));
				if (abs(dist - r) < t)
				{
					inliers++;
				}
			}

			if (inliers > maxInliers)
			{
				maxInliers = inliers;
				best_center = Point2f(cx, cy);
				best_radius = r;
			}
		}

		Mat dst;
		cvtColor(src, dst, COLOR_GRAY2BGR);

		if (maxInliers > 0)
		{
			circle(dst, best_center, best_radius, Scalar(0, 0, 255), 2);
		}

		imshow("RANSAC Circle Detection", dst);
		waitKey();
	}
}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	system("cls");
	destroyAllWindows();

	RANSAC_DeterminantCircle();

	return 0;
}