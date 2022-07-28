
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
        pointCloud_->normals_.assign(pointCloud_->normals_.size(), Eigen::Vector3d(0.0, 0.0, 0.0));
    }

    if (flag == "random") normalRandomInit();
    else if (flag == "estimate") normalEstimate();
    else if (flag == "visibility") normalVisibilityInit();

    visualize(pointCloud_);
    
}

void iPSR::normalRandomInit() {

    for (auto itr = pointCloud_->normals_.begin(); itr != pointCloud_->normals_.end(); itr++) {
        *itr = Eigen::Vector3d::Random();
    }
    pointCloud_->NormalizeNormals();

}

void iPSR::normalEstimate() {

    std::vector<bool> point_visibility(pointCloud_->points_.size(), false);

    // Estimate normals
    pointCloud_->EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(20));
    // pointCloud_->OrientNormalsConsistentTangentPlane(20);

    // Calculate bonding box
    open3d::geometry::AxisAlignedBoundingBox box = pointCloud_->GetAxisAlignedBoundingBox();
    std::vector<Eigen::Vector3d> boxPoints = box.GetBoxPoints();

    // Add 6 face center
    boxPoints.push_back((boxPoints[1] + boxPoints[2]) / 2);
    boxPoints.push_back((boxPoints[1] + boxPoints[3]) / 2);
    boxPoints.push_back((boxPoints[1] + boxPoints[4]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[0]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[7]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[6]) / 2);

    // Calculate radius
    double radius = (box.max_bound_ - box.min_bound_).squaredNorm();
    radius = 100 * std::sqrt(radius);

    // Estimate normal orientation from visibility
    for (Eigen::Vector3d& boxPoint : boxPoints) {
        auto mesh_indices = pointCloud_->HiddenPointRemoval(
            (boxPoint - box.GetCenter()) * 3 + box.GetCenter(),
            radius
        );
        std::vector<size_t> indices = std::get<1>(mesh_indices);
        for (auto index : indices) {
            point_visibility[index] = true;
            if (pointCloud_->normals_[index].dot(boxPoint - box.GetCenter()) < 0.0)
                pointCloud_->normals_[index] = -pointCloud_->normals_[index];
        }
    }

    // Random initialize invisible normal
    for (int i = 0; i < pointCloud_->normals_.size(); i++) {
        if (!point_visibility[i])
            pointCloud_->normals_[i] = Eigen::Vector3d::Random();
    }
    pointCloud_->NormalizeNormals();

}

void iPSR::normalVisibilityInit() {

    // Calculate bonding box
    open3d::geometry::AxisAlignedBoundingBox box = pointCloud_->GetAxisAlignedBoundingBox();
    std::vector<Eigen::Vector3d> boxPoints = box.GetBoxPoints();

    // Add 6 face center
    boxPoints.push_back((boxPoints[1] + boxPoints[2]) / 2);
    boxPoints.push_back((boxPoints[1] + boxPoints[3]) / 2);
    boxPoints.push_back((boxPoints[1] + boxPoints[4]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[0]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[7]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[6]) / 2);

    // Add 12 edge center
    boxPoints.push_back((boxPoints[0] + boxPoints[1]) / 2);
    boxPoints.push_back((boxPoints[0] + boxPoints[2]) / 2);
    boxPoints.push_back((boxPoints[1] + boxPoints[7]) / 2);
    boxPoints.push_back((boxPoints[2] + boxPoints[7]) / 2);
    boxPoints.push_back((boxPoints[4] + boxPoints[7]) / 2);
    boxPoints.push_back((boxPoints[5] + boxPoints[2]) / 2);
    boxPoints.push_back((boxPoints[3] + boxPoints[0]) / 2);
    boxPoints.push_back((boxPoints[6] + boxPoints[1]) / 2);
    boxPoints.push_back((boxPoints[4] + boxPoints[6]) / 2);
    boxPoints.push_back((boxPoints[3] + boxPoints[6]) / 2);
    boxPoints.push_back((boxPoints[4] + boxPoints[5]) / 2);
    boxPoints.push_back((boxPoints[3] + boxPoints[3]) / 2);

    // Calculate radius
    double radius = (box.max_bound_ - box.min_bound_).squaredNorm();
    radius = 100 * std::sqrt(radius);

    // Estimate normal from visibility
    for (Eigen::Vector3d& boxPoint : boxPoints) {
        auto mesh_indices = pointCloud_->HiddenPointRemoval(
            (boxPoint - box.GetCenter()) * 3 + box.GetCenter(),
            radius
        );
        std::vector<size_t> indices = std::get<1>(mesh_indices);
        for (auto index : indices)
            pointCloud_->normals_[index] += (boxPoint - box.GetCenter()).normalized();
    }

    // Random initialize invisible normal
    for (auto normal = pointCloud_->normals_.begin(); normal != pointCloud_->normals_.end(); normal++) {
        if (normal->squaredNorm() < 1e-4)
            *normal = Eigen::Vector3d::Random();
    }
    pointCloud_->NormalizeNormals();

}

std::vector<size_t> iPSR::hiddenPointRemoval(Eigen::Vector3d camera) {

    // Calculate radius
    double radius = -1.0;
    open3d::geometry::PointCloud projectedPointCloud(pointCloud_->points_);
    for (int i = 0; i < projectedPointCloud.points_.size(); i++) {
        projectedPointCloud.points_[i] -= camera;
        double length = projectedPointCloud.points_[i].squaredNorm();
        if (length > radius)
            radius = length;
    }
    radius = std::sqrt(radius);

    // Perform spherical projection
    for (int i = 0; i < projectedPointCloud.points_.size(); i++) {
        double norm = projectedPointCloud.points_[i].squaredNorm();
        norm = std::sqrt(norm);
        if (norm == 0.0) norm = 0.0001;
        projectedPointCloud.points_[i] += + 2 * (radius - norm) * projectedPointCloud.points_[i] / norm;
    }

    // Add camera position
    projectedPointCloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Compute convex hull
    auto mesh_indices = projectedPointCloud.ComputeConvexHull();
    return std::get<1>(mesh_indices);

}

std::shared_ptr<open3d::geometry::TriangleMesh> iPSR::execute() {

    if (!pointCloud_) {
        return nullptr;
    }

    std::shared_ptr<open3d::geometry::TriangleMesh> resultMesh;

    // Random normal initialization
    if (!pointCloud_->HasNormals()) normalRandomInit();
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
            //faceNormals[i].normalize();
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