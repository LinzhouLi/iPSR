#pragma once

#include <queue>
#include <algorithm>
#include <iostream>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <list>

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "open3d/Open3D.h"


// clang-format off
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4701 4703 4245 4189)
// 4701: potentially uninitialized local variable
// 4703: potentially uninitialized local pointer variable
// 4245: signed/unsigned mismatch
// 4189: local variable is initialized but not referenced
#endif
#include "psr/PreProcessor.h"
#include "psr/FEMTree.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
// clang-format on



// The order of the B-Spline used to splat in data for color interpolation
static const int DATA_DEGREE = 2;
// The order of the B-Spline used to splat in the weights for density estimation
static const int WEIGHT_DEGREE = 2;
// The order of the B-Spline used to splat in the normals for constructing the
// Laplacian constraints
static const int NORMAL_DEGREE = 2;
// The default finite-element degree
static const int DEFAULT_FEM_DEGREE = 1;
// The default finite-element boundary type
static const BoundaryType DEFAULT_FEM_BOUNDARY = BOUNDARY_DIRICHLET;
// The dimension of the system
static const int DIMENSION = 3;


class Open3DData {
public:
    Open3DData() : normal_(0, 0, 0), color_(0, 0, 0) {}
    Open3DData(const Eigen::Vector3d& normal, const Eigen::Vector3d& color)
        : normal_(normal), color_(color) {}

    Open3DData operator*(double s) const {
        return Open3DData(s * normal_, s * color_);
    }
    Open3DData operator/(double s) const {
        return Open3DData(normal_ / s, (1 / s) * color_);
    }
    Open3DData& operator+=(const Open3DData& d) {
        normal_ += d.normal_;
        color_ += d.color_;
        return *this;
    }
    Open3DData& operator*=(double s) {
        normal_ *= s;
        color_ *= s;
        return *this;
    }

public:
    Eigen::Vector3d normal_;
    Eigen::Vector3d color_;
};

template <typename Real>
class Open3DPointStream
    : public InputPointStreamWithData<Real, DIMENSION, Open3DData> {
public:
    Open3DPointStream(const std::shared_ptr<open3d::geometry::PointCloud> pcd)
        : pcd_(pcd), xform_(nullptr), current_(0) {}
    void reset(void) { current_ = 0; }
    bool nextPoint(Point<Real, 3>& p, Open3DData& d) {
        if (current_ >= pcd_->points_.size()) {
            return false;
        }
        p.coords[0] = static_cast<Real>(pcd_->points_[current_](0));
        p.coords[1] = static_cast<Real>(pcd_->points_[current_](1));
        p.coords[2] = static_cast<Real>(pcd_->points_[current_](2));

        if (xform_ != nullptr) {
            p = (*xform_) * p;
        }

        if (pcd_->HasNormals()) {
            d.normal_ = pcd_->normals_[current_];
        }
        else {
            d.normal_ = Eigen::Vector3d(0, 0, 0);
        }

        if (pcd_->HasColors()) {
            d.color_ = pcd_->colors_[current_];
        }
        else {
            d.color_ = Eigen::Vector3d(0, 0, 0);
        }

        current_++;
        return true;
    }

public:
    const std::shared_ptr<open3d::geometry::PointCloud> pcd_;
    XForm<Real, 4>* xform_;
    size_t current_;
};

template <typename _Real>
class Open3DVertex {
public:
    typedef _Real Real;

    Open3DVertex() : Open3DVertex(Point<Real, 3>(0, 0, 0)) {}
    Open3DVertex(Point<Real, 3> point)
        : point(point), normal_(0, 0, 0), color_(0, 0, 0), w_(0) {}

    Open3DVertex& operator*=(Real s) {
        point *= s;
        normal_ *= s;
        color_ *= s;
        w_ *= s;
        return *this;
    }

    Open3DVertex& operator+=(const Open3DVertex& p) {
        point += p.point;
        normal_ += p.normal_;
        color_ += p.color_;
        w_ += p.w_;
        return *this;
    }

    Open3DVertex& operator/=(Real s) {
        point /= s;
        normal_ /= s;
        color_ /= s;
        w_ /= s;
        return *this;
    }

public:
    // point can not have trailing _, because template methods assume that it is
    // named this way
    Point<Real, 3> point;
    Eigen::Vector3d normal_;
    Eigen::Vector3d color_;
    double w_;
};

template <unsigned int Dim, class Real>
struct FEMTreeProfiler {
    FEMTree<Dim, Real>& tree;
    double t;

    FEMTreeProfiler(FEMTree<Dim, Real>& tree) : tree(tree), t(0.0) {}
    void start(void) {
        t = Time(), FEMTree<Dim, Real>::ResetLocalMemoryUsage();
    }
    void dumpOutput(const char* header) const {
        FEMTree<Dim, Real>::MemoryUsage();
        if (header) {
            open3d::utility::LogDebug("{} {} (s), {} (MB) / {} (MB) / {} (MB)", header,
                Time() - t,
                FEMTree<Dim, Real>::LocalMemoryUsage(),
                FEMTree<Dim, Real>::MaxMemoryUsage(),
                MemoryInfo::PeakMemoryUsageMB());
        }
        else {
            open3d::utility::LogDebug("{} (s), {} (MB) / {} (MB) / {} (MB)", Time() - t,
                FEMTree<Dim, Real>::LocalMemoryUsage(),
                FEMTree<Dim, Real>::MaxMemoryUsage(),
                MemoryInfo::PeakMemoryUsageMB());
        }
    }
};

template <unsigned int Dim, typename Real>
struct ConstraintDual {
    Real target, weight;
    ConstraintDual(Real t, Real w) : target(t), weight(w) {}
    CumulativeDerivativeValues<Real, Dim, 0> operator()(
        const Point<Real, Dim>& p) const {
        return CumulativeDerivativeValues<Real, Dim, 0>(target * weight);
    };
};

template <unsigned int Dim, typename Real>
struct SystemDual {
    Real weight;
    SystemDual(Real w) : weight(w) {}
    CumulativeDerivativeValues<Real, Dim, 0> operator()(
        const Point<Real, Dim>& p,
        const CumulativeDerivativeValues<Real, Dim, 0>& dValues) const {
        return dValues * weight;
    };
    CumulativeDerivativeValues<double, Dim, 0> operator()(
        const Point<Real, Dim>& p,
        const CumulativeDerivativeValues<double, Dim, 0>& dValues) const {
        return dValues * weight;
    };
};

template <unsigned int Dim>
struct SystemDual<Dim, double> {
    typedef double Real;
    Real weight;
    SystemDual(Real w) : weight(w) {}
    CumulativeDerivativeValues<Real, Dim, 0> operator()(
        const Point<Real, Dim>& p,
        const CumulativeDerivativeValues<Real, Dim, 0>& dValues) const {
        return dValues * weight;
    };
};


class iPSR {
public:
    iPSR(int iter_num = 20, int k_neighbor = 10, int depth = 8) :
        iter_num_(iter_num),
        k_neighbor_(k_neighbor),
        depth_(depth),
        visualizer_(nullptr),
        pointCloud_(nullptr),
        scale_(1.1f)
    { }

    void normalInit(const std::string& flag = "random") {

        if (!pointCloud_) {
            return;
        }

        pointCloud_->normals_.resize(pointCloud_->points_.size());
        pointCloud_->normals_.assign(pointCloud_->normals_.size(), Eigen::Vector3d(0.0, 0.0, 0.0));

        if (flag == "estimate") normalEstimate();
        else if (flag == "visibility") normalVisibilityInit();
        else normalRandomInit();

    }
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

    std::shared_ptr<open3d::geometry::TriangleMesh> execute();

private:
    int iter_num_, k_neighbor_, depth_;
    float scale_;
    std::shared_ptr<open3d::visualization::Visualizer> visualizer_;
    std::shared_ptr<open3d::geometry::PointCloud> pointCloud_;

    void normalRandomInit();
    void normalEstimate();
    void normalVisibilityInit();
    std::vector<size_t> hiddenPointRemoval(Eigen::Vector3d camera);

    void visualize(std::shared_ptr<open3d::geometry::Geometry> geometry) {
        if (visualizer_ && geometry) {
            visualizer_->ClearGeometries();
            visualizer_->AddGeometry(geometry);
            visualizer_->PollEvents();
            visualizer_->UpdateRender();
        }
    }

    template <class Real, unsigned int Dim>
    void _execute(std::shared_ptr<open3d::geometry::TriangleMesh> out_mesh);

    template <unsigned int Dim, typename Real>
    void octreeSampePoints(
        Open3DPointStream<Real>& pointStream,
        std::vector<Real>& sample_weight
    );

    template <typename Real, unsigned int Dim, unsigned int... FEMSigs>
    void poissonReconstruction(
        std::shared_ptr<open3d::geometry::TriangleMesh>& out_mesh,
        const XForm<Real, Dim + 1>& iXForm,
        const std::vector<Real>& sample_weight,
        UIntPack<FEMSigs...>
    );

};





//************************** PSR utils **************************

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetBoundingBoxXForm(
    Point<Real, Dim> min,
    Point<Real, Dim> max,
    Real scaleFactor
) { // = 1.1
    Point<Real, Dim> center = (max + min) / 2;
    Real scale = max[0] - min[0];
    for (unsigned int d = 1; d < Dim; d++) {
        scale = std::max<Real>(scale, max[d] - min[d]);
    }
    scale *= scaleFactor;
    for (unsigned int i = 0; i < Dim; i++) {
        center[i] -= scale / 2;
    }
    XForm<Real, Dim + 1> tXForm = XForm<Real, Dim + 1>::Identity(),
        sXForm = XForm<Real, Dim + 1>::Identity();
    for (unsigned int i = 0; i < Dim; i++) {
        sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
    }
    return sXForm * tXForm;
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetBoundingBoxXForm(
    Point<Real, Dim> min,
    Point<Real, Dim> max,
    Real width,
    Real scaleFactor,
    int& depth
) {
    // Get the target resolution (along the largest dimension)
    Real resolution = (max[0] - min[0]) / width;
    for (unsigned int d = 1; d < Dim; d++) {
        resolution = std::max<Real>(resolution, (max[d] - min[d]) / width);
    }
    resolution *= scaleFactor;
    depth = 0;
    while ((1 << depth) < resolution) {
        depth++;
    }

    Point<Real, Dim> center = (max + min) / 2;
    Real scale = (1 << depth) * width;

    for (unsigned int i = 0; i < Dim; i++) {
        center[i] -= scale / 2;
    }
    XForm<Real, Dim + 1> tXForm = XForm<Real, Dim + 1>::Identity(),
        sXForm = XForm<Real, Dim + 1>::Identity();
    for (unsigned int i = 0; i < Dim; i++) {
        sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
    }
    return sXForm * tXForm;
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetPointXForm(
    InputPointStream<Real, Dim>& stream,
    Real width,
    Real scaleFactor,
    int& depth
) {
    Point<Real, Dim> min, max;
    stream.boundingBox(min, max);
    return GetBoundingBoxXForm(min, max, width, scaleFactor, depth);
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetPointXForm(
    InputPointStream<Real, Dim>& stream,
    Real scaleFactor
) {
    Point<Real, Dim> min, max;
    stream.boundingBox(min, max);
    return GetBoundingBoxXForm(min, max, scaleFactor);
}


//************************** iPSR normal init **************************

void iPSR::normalRandomInit() {

    ThreadPool::Parallel_for(
        0, pointCloud_->normals_.size(),
        [&](unsigned int thread, size_t i) {
            pointCloud_->normals_[i] = Eigen::Vector3d::Random();
        }
    );
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
        projectedPointCloud.points_[i] += +2 * (radius - norm) * projectedPointCloud.points_[i] / norm;
    }

    // Add camera position
    projectedPointCloud.points_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Compute convex hull
    auto mesh_indices = projectedPointCloud.ComputeConvexHull();
    return std::get<1>(mesh_indices);

}

//************************** iPSR execute **************************

template <unsigned int Dim, typename Real>
void iPSR::octreeSampePoints(
    Open3DPointStream<Real>& pointStream,
    std::vector<Real>& sample_weight
) {

    FEMTree<Dim, Real> octree(MEMORY_ALLOCATOR_BLOCK_SIZE);
    std::vector<typename FEMTree<Dim, Real>::PointSample> samples; // coordinate
    std::vector<Open3DData> sampleData; // normal ( and color )

    // octree sample points
    FEMTreeInitializer<Dim, Real>::Initialize<Open3DData>(
        octree.spaceRoot(),
        pointStream,
        depth_,
        samples,
        sampleData,
        true,
        octree.nodeAllocators[0],
        octree.initializer(),
        [](const Point<Real, Dim>& p, Open3DData& d) {
            Real l = (Real)d.normal_.norm();
            if (!l || l != l) return (Real)-1.;
            d.normal_ /= l;
            return (Real)1.;
        }
    );

    // update point cloud
    pointCloud_->points_.resize(samples.size());
    pointCloud_->normals_.resize(samples.size());
    sample_weight.resize(samples.size());
    ThreadPool::Parallel_for(
        0, samples.size(),
        [&](unsigned int thread, size_t i) {
            Real weight = samples[i].sample.weight;
            sample_weight[i] = weight;
            pointCloud_->points_[i][0] = samples[i].sample.data.coords[0] / weight;
            pointCloud_->points_[i][1] = samples[i].sample.data.coords[1] / weight;
            pointCloud_->points_[i][2] = samples[i].sample.data.coords[2] / weight;
            pointCloud_->normals_[i] = sampleData[i].normal_ / weight;
        }
    );

}


template <typename Vertex, typename Real, typename SetVertexFunction, unsigned int... FEMSigs>
void ExtractMesh(
    float datax,
    bool linear_fit,
    UIntPack<FEMSigs...>,
    FEMTree<sizeof...(FEMSigs), Real>& tree,
    const DenseNodeData<Real, UIntPack<FEMSigs...>>& solution,
    Real isoValue,
    const std::vector<typename FEMTree<sizeof...(FEMSigs),
    Real>::PointSample>* samples,
    std::vector<Open3DData>* sampleData,
    const typename FEMTree<sizeof...(FEMSigs),
    Real>::template DensityEstimator<WEIGHT_DEGREE>*
    density,
    const SetVertexFunction& SetVertex,
    XForm<Real, sizeof...(FEMSigs) + 1> iXForm,
    std::shared_ptr<open3d::geometry::TriangleMesh>& out_mesh,
    std::vector<double>& out_densities
) {
    static const int Dim = sizeof...(FEMSigs);
    typedef UIntPack<FEMSigs...> Sigs;
    static const unsigned int DataSig = FEMDegreeAndBType<DATA_DEGREE, DEFAULT_FEM_BOUNDARY>::Signature;
    typedef typename FEMTree<Dim, Real>::template DensityEstimator<WEIGHT_DEGREE> DensityEstimator;

    FEMTreeProfiler<Dim, Real> profiler(tree);

    CoredMeshData<Vertex, node_index_type>* mesh;
    mesh = new CoredVectorMeshData<Vertex, node_index_type>();

    bool non_manifold = true;
    bool polygon_mesh = false;

    profiler.start();
    typename IsoSurfaceExtractor<Dim, Real, Vertex>::IsoStats isoStats;
    if (sampleData) {
        SparseNodeData<ProjectiveData<Open3DData, Real>, IsotropicUIntPack<Dim, DataSig>>
            _sampleData = tree.template setMultiDepthDataField<DataSig, false>(*samples, *sampleData, (DensityEstimator*)NULL);
        for (
            const RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type>* n = tree.tree().nextNode();
            n;
            n = tree.tree().nextNode(n)
        ) {
            ProjectiveData<Open3DData, Real>* clr = _sampleData(n);
            if (clr) (*clr) *= (Real)pow(datax, tree.depth(n));
        }
        isoStats = IsoSurfaceExtractor<Dim, Real, Vertex>::template Extract<Open3DData>(
            Sigs(), UIntPack<WEIGHT_DEGREE>(),
            UIntPack<DataSig>(), tree, density, &_sampleData,
            solution, isoValue, *mesh, SetVertex, !linear_fit,
            !non_manifold, polygon_mesh, false
        );
    }
    else {
        isoStats = IsoSurfaceExtractor<Dim, Real, Vertex>::template Extract<Open3DData>(
            Sigs(), UIntPack<WEIGHT_DEGREE>(),
            UIntPack<DataSig>(), tree, density, NULL, solution,
            isoValue, *mesh, SetVertex, !linear_fit,
            !non_manifold, polygon_mesh, false
        );
    }
    
    mesh->resetIterator();
    out_densities.clear();
    out_mesh->vertices_.clear();
    out_mesh->vertex_normals_.clear();
    out_mesh->vertex_colors_.clear();
    out_mesh->triangles_.clear();
    for (size_t vidx = 0; vidx < mesh->outOfCorePointCount(); ++vidx) {
        Vertex v;
        mesh->nextOutOfCorePoint(v);
        v.point = iXForm * v.point;
        out_mesh->vertices_.push_back(Eigen::Vector3d(v.point[0], v.point[1], v.point[2]));
        out_mesh->vertex_normals_.push_back(v.normal_);
        out_mesh->vertex_colors_.push_back(v.color_);
        out_densities.push_back(v.w_);
    }
    for (size_t tidx = 0; tidx < mesh->polygonCount(); ++tidx) {
        std::vector<CoredVertexIndex<node_index_type>> triangle;
        mesh->nextPolygon(triangle);
        if (triangle.size() != 3) {
            open3d::utility::LogError("got polygon");
        }
        else {
            out_mesh->triangles_.push_back(Eigen::Vector3i(triangle[0].idx, triangle[1].idx, triangle[2].idx));
        }
    }

    delete mesh;
}


template <typename Real, unsigned int Dim, unsigned int... FEMSigs>
void iPSR::poissonReconstruction(
    std::shared_ptr<open3d::geometry::TriangleMesh>& out_mesh,
    const XForm<Real, Dim + 1>& iXForm,
    const std::vector<Real>& sample_weight,
    UIntPack<FEMSigs...>
) {

    using Sigs = UIntPack<FEMSigs...>;
    using Degrees = UIntPack<FEMSignature<FEMSigs>::Degree...>;
    using InterpolationInfo = typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 0>;
    using DensityEstimator = typename FEMTree<Dim, Real>::template DensityEstimator<WEIGHT_DEGREE>;
    using NormalSigs = UIntPack<FEMDegreeAndBType<NORMAL_DEGREE, DerivativeBoundary<FEMSignature<FEMSigs>::BType, 1>::BType>::Signature...>;

    float datax = 32.f;
    int base_depth = 0;
    int base_v_cycles = 1;
    float point_weight = 10.f;
    float samples_per_node = 1.5f; // 
    float cg_solver_accuracy = 1e-3f;
    int full_depth = 5;
    int iters = 8;
    bool exact_interpolation = false;

    FEMTree<Dim, Real> tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
    FEMTreeProfiler<Dim, Real> profiler(tree);

    size_t pointCount;
    std::vector<typename FEMTree<Dim, Real>::PointSample> samples; // coordinate
    std::vector<Open3DData> sampleData; // normal ( and color )
    Real isoValue = (Real)0;
    Real targetValue = (Real)0.5;
    Real pointWeightSum;
    DensityEstimator* density = NULL;
    SparseNodeData<Point<Real, Dim>, NormalSigs>* normalInfo = NULL;

    int kernelDepth = depth_ - 2;
    if (kernelDepth < 0) {
        open3d::utility::LogError("depth (={}) has to be >= 2", depth_);
    }
    
    // Sample points
    {
        Open3DPointStream<Real> pointStream(pointCloud_);
        pointCount = FEMTreeInitializer<Dim, Real>::Initialize<Open3DData>(
            tree.spaceRoot(),
            pointStream,
            depth_,
            samples,
            sampleData,
            true,
            tree.nodeAllocators[0],
            tree.initializer(),
            [](const Point<Real, Dim>& p, Open3DData& d) {
                Real l = (Real)d.normal_.norm();
                if (!l || l != l) return (Real)-1.;
                d.normal_ /= l;
                return (Real)1.;
            }
        );
        ThreadPool::Parallel_for(
            0, samples.size(),
            [&](unsigned int thread, size_t i) {
                Real weight = sample_weight[i];
                samples[i].sample.weight = weight;
                samples[i].sample.data *= weight;
            }
        );
    }
    
    DenseNodeData<Real, Sigs> solution;
    {
        DenseNodeData<Real, Sigs> constraints;
        InterpolationInfo* iInfo = NULL;
        int solveDepth = depth_;

        tree.resetNodeIndices();

        // Get the kernel density estimator
        {
            profiler.start();
            density = tree.template setDensityEstimator<WEIGHT_DEGREE>(samples, kernelDepth, samples_per_node, 1);
            profiler.dumpOutput("#   Got kernel density:");
        }
        
        // Transform the Hermite samples into a vector field
        {
            profiler.start();
            normalInfo = new SparseNodeData<Point<Real, Dim>, NormalSigs>();
            std::function<bool(Open3DData, Point<Real, Dim>&)>
                ConversionFunction = [](Open3DData in, Point<Real, Dim>& out) { // Open3DData => Normal
                Point<Real, Dim> n(in.normal_(0), in.normal_(1), in.normal_(2));
                Real l = (Real)Length(n);
                if (!l) return false; // It is possible that the samples have non-zero normals
                out = n / l;
                return true;
            };
            *normalInfo = tree.setDataField(
                NormalSigs(), samples, sampleData, density,
                pointWeightSum, ConversionFunction
            );
            ThreadPool::Parallel_for( // normal = -normal
                0, normalInfo->size(),
                [&](unsigned int, size_t i) { (*normalInfo)[i] *= (Real)-1.; }
            );
            profiler.dumpOutput("#     Got normal field:");
            open3d::utility::LogDebug("Point weight / Estimated Area: {:e} / {:e}", pointWeightSum, pointCount * pointWeightSum);
        }
        
        // Trim the tree and prepare for multigrid
        {
            profiler.start();
            constexpr int MAX_DEGREE = NORMAL_DEGREE > Degrees::Max() ? NORMAL_DEGREE : Degrees::Max();
            tree.template finalizeForMultigrid<MAX_DEGREE>(
                full_depth,
                typename FEMTree<Dim, Real>::template HasNormalDataFunctor<NormalSigs>(*normalInfo),
                normalInfo,
                density
            );
            profiler.dumpOutput("#       Finalized tree:");
        }
        
        // Add the FEM constraints
        {
            profiler.start();
            constraints = tree.initDenseNodeData(Sigs());
            typename FEMIntegrator::template Constraint<
                Sigs,
                IsotropicUIntPack<Dim, 1>,
                NormalSigs,
                IsotropicUIntPack<Dim, 0>,
                Dim
            > F;
            unsigned int derivatives2[Dim];
            for (unsigned int d = 0; d < Dim; d++) derivatives2[d] = 0;
            using Derivatives1 = IsotropicUIntPack<Dim, 1>;
            using Derivatives2 = IsotropicUIntPack<Dim, 0>;
            for (unsigned int d = 0; d < Dim; d++) {
                unsigned int derivatives1[Dim];
                for (unsigned int dd = 0; dd < Dim; dd++)
                    derivatives1[dd] = dd == d ? 1 : 0;
                F.weights
                    [d]
                    [TensorDerivatives<Derivatives1>::Index(derivatives1)]
                    [TensorDerivatives<Derivatives2>::Index(derivatives2)]
                = 1;
            }
            tree.addFEMConstraints(F, *normalInfo, constraints, solveDepth);
            profiler.dumpOutput("#  Set FEM constraints:");
        }
        
        // Free up the normal info
        delete normalInfo, normalInfo = NULL;
        
        // Add the interpolation constraints
        if (point_weight > 0) {
            profiler.start();
            if (exact_interpolation) {
                iInfo = FEMTree<Dim, Real>::template InitializeExactPointInterpolationInfo<Real, 0>(
                    tree,
                    samples,
                    ConstraintDual<Dim, Real>(targetValue, (Real)point_weight * pointWeightSum),
                    SystemDual<Dim, Real>((Real)point_weight * pointWeightSum),
                    true,
                    false
                );
            }
            else {
                iInfo = FEMTree<Dim, Real>::template InitializeApproximatePointInterpolationInfo<Real, 0>(
                    tree,
                    samples,
                    ConstraintDual<Dim, Real>(targetValue, (Real)point_weight * pointWeightSum),
                    SystemDual<Dim, Real>((Real)point_weight * pointWeightSum),
                    true,
                    1
                );
            }
            tree.addInterpolationConstraints(constraints, solveDepth, *iInfo);
            profiler.dumpOutput("#Set point constraints:");
        }
        
        open3d::utility::LogDebug("Leaf Nodes / Active Nodes / Ghost Nodes: {} / {} / {}", tree.leaves(), tree.nodes(), tree.ghostNodes());
        open3d::utility::LogDebug("Memory Usage: {:.3f} MB", float(MemoryInfo::Usage()) / (1 << 20));

        // Solve the linear system
        {
            profiler.start();
            typename FEMTree<Dim, Real>::SolverInfo sInfo;
            sInfo.cgDepth = 0,
                sInfo.cascadic = true, 
                sInfo.vCycles = 1,
                sInfo.iters = iters, 
                sInfo.cgAccuracy = cg_solver_accuracy,
                sInfo.verbose = open3d::utility::GetVerbosityLevel() == open3d::utility::VerbosityLevel::Debug,
                sInfo.showResidual = open3d::utility::GetVerbosityLevel() == open3d::utility::VerbosityLevel::Debug,
                sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE,
                sInfo.sliceBlockSize = 1,
                sInfo.baseDepth = base_depth,
                sInfo.baseVCycles = base_v_cycles;
            typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim, 1>> F({ 0., 1. });
            solution = tree.solveSystem(Sigs(), F, constraints, solveDepth, sInfo, iInfo);
            profiler.dumpOutput("# Linear system solved:");
            if (iInfo) delete iInfo, iInfo = NULL;
        }
    }
    
    // caculate iso value
    {
        profiler.start();
        double valueSum = 0, weightSum = 0;
        typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 0> evaluator(&tree, solution);
        std::vector<double>
            valueSums(ThreadPool::NumThreads(), 0),
            weightSums(ThreadPool::NumThreads(), 0);
        ThreadPool::Parallel_for(
            0, samples.size(),
            [&](unsigned int thread, size_t j) {
                ProjectiveData<Point<Real, Dim>, Real>& sample = samples[j].sample;
                Real w = sample.weight;
                if (w > 0)
                    weightSums[thread] += w,
                    valueSums[thread] += evaluator.values(sample.data / sample.weight, thread, samples[j].node)[0] * w;
            }
        );
        for (size_t t = 0; t < valueSums.size(); t++)
            valueSum += valueSums[t],
            weightSum += weightSums[t];
        isoValue = (Real)(valueSum / weightSum);
        profiler.dumpOutput("Got average:");
        open3d::utility::LogDebug("Iso-Value: {:e} = {:e} / {:e}", isoValue, valueSum, weightSum);
    }
    
    auto SetVertex = [](Open3DVertex<Real>& v, Point<Real, Dim> p, Real w, Open3DData d) {
        v.point = p;
        v.normal_ = d.normal_;
        v.color_ = d.color_;
        v.w_ = w;
    };
    std::vector<double> out_densities;
    ExtractMesh<Open3DVertex<Real>, Real>(
        datax, false, UIntPack<FEMSigs...>(),
        tree, solution, isoValue, &samples,
        &sampleData, density, SetVertex, iXForm, out_mesh, out_densities
    );

    if (density) delete density, density = NULL;
    
}


template <typename Real, unsigned int Dim>
void iPSR::_execute(
    std::shared_ptr<open3d::geometry::TriangleMesh> out_mesh
) {

    using FEMSigs = IsotropicUIntPack<Dim, FEMDegreeAndBType<DATA_DEGREE, DEFAULT_FEM_BOUNDARY>::Signature>;

    double startTime = Time();
    XForm<Real, Dim + 1> xForm, iXForm;
    xForm = XForm<Real, Dim + 1>::Identity();
    std::vector<Real> sample_weight;
    auto resultMesh = std::make_shared<open3d::geometry::TriangleMesh>();

    // Random normal initialization
    if (!pointCloud_->HasNormals()) normalRandomInit();

    // sample points with octree
    {
        Open3DPointStream<Real> pointStream(pointCloud_);
        xForm = GetPointXForm<Real, Dim>(pointStream, (Real)scale_) * xForm;
        iXForm = xForm.inverse();
        pointStream.xform_ = &xForm;
        octreeSampePoints<Dim, Real>(pointStream, sample_weight);
    }

    // Build kd-tree
    open3d::geometry::KDTreeFlann kdTree(*pointCloud_);
    //visualize(pointCloud_);

    // iPSR iterations
    for (int epoch = 0; epoch < 1; epoch++) {
        std::cout << "Iter: " << epoch << std::endl;
        std::vector<Eigen::Vector3d> originNormals(pointCloud_->normals_);

        // Poisson surface reconstruction
        poissonReconstruction<Real, Dim>(
            resultMesh,
            XForm<Real, Dim + 1>::Identity(),
            sample_weight,
            FEMSigs()
        );
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
        }
        
        // Update sample point normals
        pointCloud_->normals_.assign(pointCloud_->normals_.size(), Eigen::Vector3d(1e-4, 1e-4, 1e-4));
        for (int i = 0; i < kNeighborSamplePoints.size(); i++) {
            for (int j = 0; j < kNeighborSamplePoints[i].size(); j++) {
                pointCloud_->normals_[kNeighborSamplePoints[i][j]] += faceNormals[i];
            }
        }
        pointCloud_->NormalizeNormals();
        
        // Compute the average normal variation of the top 0.1% points
        std::vector<double> diff(pointCloud_->normals_.size());
        ThreadPool::Parallel_for(
            0, pointCloud_->normals_.size(),
            [&](unsigned int thread, size_t i) {
                diff[i] = (pointCloud_->normals_[i] - originNormals[i]).squaredNorm();
            }
        );
        
        size_t heap_size = static_cast<size_t>(ceil(pointCloud_->normals_.size() / 1000.0));
        std::priority_queue<double, std::vector<double>, std::greater<double>> min_heap;
        for (int i = 0; i < diff.size(); i++) {
            if (min_heap.size() < heap_size) {
                min_heap.push(diff[i]);
            }
            else if (diff[i] > min_heap.top()) {
                min_heap.pop();
                min_heap.push(diff[i]);
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
    poissonReconstruction<Real, Dim>(
        resultMesh,
        iXForm,
        sample_weight,
        FEMSigs()
    );

}


std::shared_ptr<open3d::geometry::TriangleMesh> iPSR::execute() {

    if (!pointCloud_) {
        return nullptr;
    }

    int n_threads = (int)std::thread::hardware_concurrency();

#ifdef _OPENMP
    ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP,
        n_threads);
#else
    ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL,
        n_threads);
#endif

    std::shared_ptr<open3d::geometry::TriangleMesh> resultMesh;
    _execute<double, DIMENSION>(resultMesh);

    ThreadPool::Terminate();

    return resultMesh;

}
