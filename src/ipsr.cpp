
#include <queue>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "ipsr.h"

void iPSR::normalInit(const std::string& flag = "random") {

    if (!pointCloud_) {
        return;
    }
    if (!pointCloud_->HasNormals()) {
        pointCloud_->normals_.resize(pointCloud_->points_.size());
    }

    if (flag == "random") nomalRandomInit();
    
}

void iPSR::nomalRandomInit() {

    for (auto itr = pointCloud_->normals_.begin(); itr != pointCloud_->normals_.end(); itr++) {
        *itr = Eigen::Vector3d::Random();
    }
    pointCloud_->NormalizeNormals();

}

std::shared_ptr<open3d::geometry::TriangleMesh> iPSR::execute() {

    if (!pointCloud_) {
        return nullptr;
    }

    std::shared_ptr<open3d::geometry::TriangleMesh> resultMesh;

    // Random normal initialization
    if (!pointCloud_->HasNormals()) nomalRandomInit();
    visualize(pointCloud_);

    // Build kd-tree
    open3d::geometry::KDTreeFlann kdTree(*pointCloud_);

    // Iterations
    for (int epoch = 0; epoch < iter_num_; epoch++) {
        std::cout << "Iter: " << epoch << std::endl;
        std::vector<Eigen::Vector3d> originNormals(pointCloud_->normals_);

        // Poisson surface reconstruction
        auto result = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*pointCloud_, depth_, 0.0, 1.0);
        resultMesh = std::get<0>(result);
        visualize(resultMesh);

        // Compute face normals and map them to sample points
        std::vector<std::vector<int>> kNeighborSamplePoints(resultMesh->triangles_.size());
        std::vector<Eigen::Vector3d> faceNormals(resultMesh->triangles_.size());
        std::vector<double> distance2;

        for (int i = 0; i < resultMesh->triangles_.size(); i++) {
            Eigen::Vector3i triangle = resultMesh->triangles_[i];
            Eigen::Vector3d triangleCenter = (
                resultMesh->vertices_[triangle(0)] +
                resultMesh->vertices_[triangle(1)] +
                resultMesh->vertices_[triangle(2)]
            ) / 3.0;
            kdTree.SearchKNN(triangleCenter, k_neighbor_, kNeighborSamplePoints[i], distance2);
            faceNormals[i] = (resultMesh->vertices_[triangle(1)] - resultMesh->vertices_[triangle(0)])
                .cross(resultMesh->vertices_[triangle(2)] - resultMesh->vertices_[triangle(0)]);
            faceNormals[i].normalize();
        }

        // Update sample point normals
        pointCloud_->normals_.assign(pointCloud_->normals_.size(), Eigen::Vector3d(1e-4, 1e-4, 1e-4));
        for (int i = 0; i < kNeighborSamplePoints.size(); i++) {
            for (int j = 0; j < kNeighborSamplePoints[i].size(); j++) {
                pointCloud_->normals_[kNeighborSamplePoints[i][j]] += faceNormals[i];
            }
        }
        pointCloud_->NormalizeNormals();
        //visualize(pointCloud_);

        // Compute the average normal variation of the top 1/1000 points
        size_t heap_size = static_cast<size_t>(ceil(pointCloud_->normals_.size() / 1000.0));
        std::priority_queue<double, std::vector<double>, std::greater<double>> min_heap;

        for (int i = 0; i < pointCloud_->normals_.size(); i++) {
            double diff = (pointCloud_->normals_[i] - originNormals[i]).squaredNorm();
            if (min_heap.size() < heap_size) {
                min_heap.push(diff);
            }
            else if (diff > min_heap.top()) {
                min_heap.pop();
                min_heap.push(diff);
            }
        }

        heap_size = min_heap.size();
        double avg_max_diff = 0;
        while (!min_heap.empty()) {
            avg_max_diff += std::sqrt(min_heap.top());
            min_heap.pop();
        }
        avg_max_diff /= heap_size;
        std::cout << "Avg variation of top 1000 normals: " << avg_max_diff << std::endl;

        if (avg_max_diff < 0.175)
            break;
    }

    // Poisson surface reconstruction
    auto result = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*pointCloud_);
    resultMesh = std::get<0>(result);
    //visualize(pointCloud_);
    visualize(resultMesh);

    return resultMesh;

}