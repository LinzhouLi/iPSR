#pragma once

#include <string>
#include "open3d/Open3D.h"

class iPSR {
private:
    int iter_num_, k_neighbor_, depth_;
    std::shared_ptr<open3d::visualization::Visualizer> visualizer_;
    std::shared_ptr<open3d::geometry::PointCloud> pointCloud_;

    void nomalRandomInit();
    void visualize(std::shared_ptr<open3d::geometry::Geometry> geometry) {
        if (visualizer_ && geometry) {
            visualizer_->ClearGeometries();
            visualizer_->AddGeometry(geometry);
            visualizer_->PollEvents();
            visualizer_->UpdateRender();
        }
    }

public:
    iPSR(int iter_num = 80, int k_neighbor = 10, int depth = 8) : 
        iter_num_(iter_num), 
        k_neighbor_(k_neighbor),
        depth_(depth),
        visualizer_(nullptr), 
        pointCloud_(nullptr)
    { }

    void setInputPointCloud(
        std::shared_ptr<open3d::geometry::PointCloud> pointCloud) {
        pointCloud_ = pointCloud;
        //visualize(pointCloud_);
    }
    void setVisualizer(
        std::shared_ptr<open3d::visualization::Visualizer> visualizer) {
        visualizer_ = visualizer;
        //visualize(pointCloud_);
    }

    void normalInit(const std::string& flag);
    std::shared_ptr<open3d::geometry::TriangleMesh> execute();

};
