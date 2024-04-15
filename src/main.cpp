
#include <string>
#include <iostream>
#include "open3d/Open3D.h"
#include "ipsr.h"


int main(int argc, char *argv[]) {
    
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    open3d::io::ReadPointCloudFromPLY("D:/code/3DGS/progressive-gs/data/hotdog.ply", *cloud, open3d::io::ReadPointCloudOption());

    auto visualizer = std::make_shared<open3d::visualization::Visualizer>();
    visualizer->CreateVisualizerWindow("iPSR", 1280, 720, 150, 300);
    //visualizer->SetFullScreen(true);
    visualizer->GetRenderOption().point_show_normal_ = true;
    visualizer->GetRenderOption().mesh_show_wireframe_ = true;
    visualizer->GetRenderOption().mesh_color_option_ =
            open3d::visualization::RenderOption::MeshColorOption::Default;
    visualizer->GetViewControl().Translate(1000, 0);

    iPSR ipsr(30);
    ipsr.setInputPointCloud(cloud);
    ipsr.setVisualizer(visualizer);
    ipsr.normalInit("visibility");
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh = ipsr.execute();

    visualizer->Run();

    return 0;

}
