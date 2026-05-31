#ifndef HYBRID_ASTAR__A_STAR_HPP_
#define HYBRID_ASTAR__A_STAR_HPP_

#include <functional>
#include <memory>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "my/hybrid_astar/costmap_2d.hpp"

#include "robin_hood.h"
#include "my/hybrid_astar/analytic_expansion.hpp"
#include "my/hybrid_astar/node_hybrid.hpp"
#include "my/hybrid_astar/node_basic.hpp"
#include "my/hybrid_astar/goal_manager.hpp"
#include "my/hybrid_astar/types.hpp"
#include "my/hybrid_astar/constants.hpp"

namespace hybrid_astar
{

template<typename NodeT>
class AStarAlgorithm
{
public:
  typedef NodeT * NodePtr;
  typedef robin_hood::unordered_node_map<uint64_t, NodeT> Graph;
  typedef std::vector<NodePtr> NodeVector;
  typedef std::pair<float, NodeBasic<NodeT>> NodeElement;
  typedef typename NodeT::Coordinates Coordinates;
  typedef typename NodeT::CoordinateVector CoordinateVector;
  typedef typename NodeVector::iterator NeighborIterator;
  typedef std::function<bool (const uint64_t &, NodeT * &)> NodeGetter;
  typedef GoalManager<NodeT> GoalManagerT;
  using NodeContext = typename NodeT::NodeContext;

  struct NodeComparator
  {
    bool operator()(const NodeElement & a, const NodeElement & b) const
    {
      return a.first > b.first;
    }
  };

  typedef std::priority_queue<NodeElement, std::vector<NodeElement>, NodeComparator> NodeQueue;

  explicit AStarAlgorithm(const MotionModel & motion_model, const SearchInfo & search_info);
  ~AStarAlgorithm();

  void initialize(
    const bool & allow_unknown,
    int & max_iterations,
    const int & max_on_approach_iterations,
    const int & terminal_checking_interval,
    const double & max_planning_time,
    const float & lookup_table_size,
    const unsigned int & dim_3_size);

  bool createPath(
    CoordinateVector & path, int & num_iterations, const float & tolerance,
    std::function<bool()> cancel_checker,
    std::vector<std::tuple<float, float, float>> * expansions_log = nullptr);

  void setSearchInfo(const SearchInfo & search_info) {_search_info = search_info;}

  void setCollisionChecker(GridCollisionChecker * collision_checker);

  void setGoal(
    const float & mx,
    const float & my,
    const unsigned int & dim_3,
    const GoalHeadingMode & goal_heading_mode = GoalHeadingMode::DEFAULT,
    const int & coarse_search_resolution = 1);

  void setStart(
    const float & mx,
    const float & my,
    const unsigned int & dim_3);

  int & getMaxIterations();

  NodePtr & getStart();

  int & getOnApproachMaxIterations();

  float & getToleranceHeuristic();

  unsigned int & getSizeX();

  unsigned int & getSizeY();

  unsigned int & getSizeDim3();

  unsigned int getCoarseSearchResolution();

  GoalManagerT getGoalManager();

  NodeContext * getContext();

protected:
  inline NodePtr getNextNode();
  inline void addNode(const float & cost, NodePtr & node);
  inline NodePtr addToGraph(const uint64_t & index);
  inline float getHeuristicCost(const NodePtr & node);
  inline bool areInputsValid();
  inline bool getClosestPathWithinTolerance(CoordinateVector & path);
  inline void clearQueue();
  inline void clearGraph();
  inline uint64_t getIndex(
    const unsigned int & x, const unsigned int & y, const unsigned int & dim3);
  inline bool onVisitationCheckNode(const NodePtr & node);
  inline void populateExpansionsLog(
    const NodePtr & node, std::vector<std::tuple<float, float, float>> * expansions_log);

  bool _traverse_unknown;
  bool _is_initialized;
  int _max_iterations;
  int _max_on_approach_iterations;
  int _terminal_checking_interval;
  double _max_planning_time;
  float _tolerance;
  unsigned int _x_size;
  unsigned int _y_size;
  unsigned int _dim3_size;
  unsigned int _coarse_search_resolution;
  SearchInfo _search_info;

  NodePtr _start;
  GoalManagerT _goal_manager;
  Graph _graph;
  NodeQueue _queue;

  MotionModel _motion_model;
  NodeHeuristicPair _best_heuristic_node;

  GridCollisionChecker * _collision_checker;
  Costmap2D * _costmap;
  std::unique_ptr<AnalyticExpansion<NodeT>> _expander;
  std::shared_ptr<NodeContext> _shared_ctx;
};

}  // namespace hybrid_astar

#endif  // HYBRID_ASTAR__A_STAR_HPP_
