#ifndef SOAP_NPFGA_HPP
#define SOAP_NPFGA_HPP

#include <cmath>
#include <map>
#include <boost/numeric/ublas/matrix.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>    
#include "types.hpp"

namespace soap { namespace npfga {

namespace ub = boost::numeric::ublas;
namespace py = pybind11;
//namespace bpy = boost::python;
class Operator;
class FNode;
typedef double dtype_t;
typedef ub::matrix<dtype_t> matrix_t;
typedef ub::zero_matrix<dtype_t> zero_matrix_t;
typedef std::vector<FNode*> nodelist_t;

struct Instruction
{
    typedef std::vector<Instruction*> args_t;
    Instruction(Operator *oper, std::string tagstr, double pow, double prefactor);
    Instruction(Operator *op, std::vector<Instruction*> &args_in);
    Instruction() : op(NULL), tag(""), power(1.0), prefactor(1.0), is_root(false) {;}
    ~Instruction();
    // Methods
    Instruction *deepCopy(Instruction *);
    std::string getBasename();
    std::string stringify(std::string format="");
    void simplifyInstruction();
    void raiseToPower(double p);
    void multiplyBy(double c) { prefactor *= c; }
    bool containsConstant();
    // Members
    Operator *op;
    bool is_root;
    std::string tag;
    std::string expr;
    double power;
    double prefactor;
    args_t args;
};

struct FNodeStats
{
    FNodeStats() : cov(-1.), q(-1.) {;}
    ~FNodeStats() {;}
    double cov;
    double q;
};

struct FNodeDimension
{
    typedef std::map<std::string, double> dim_map_t;
    FNodeDimension() {;}
    FNodeDimension(const FNodeDimension &other) { dim_map = other.dim_map; }
    FNodeDimension(std::string dimstr);
    FNodeDimension(dim_map_t &dim_map_in);
    ~FNodeDimension() {;}
    // Methods
    std::string calculateString();
    void eraseZeros();
    void raiseToPower(double p);
    void add(FNodeDimension &other);
    void subtract(FNodeDimension &other);
    void addFactor(const std::string &unit, const double &power);
    void subtractFactor(const std::string &unit, const double &power);
    bool matches(FNodeDimension &other, bool check_reverse=true);
    bool isDimensionless() { return (dim_map.size() == 0); }
    // Members
    dim_map_t dim_map;
    // Serialization
};

struct FNodeCheck
{
    FNodeCheck(double min_power, double max_power)
        : min_pow(min_power), max_pow(max_power) {;}
    bool check(FNode* fnode);
    double min_pow;
    double max_pow;
};

class FNode
{
  public:
    FNode();
    FNode(Operator *op, std::string varname, std::string varplus,
        std::string varzero, std::string dimstr, bool is_root, double unit_prefactor);
    FNode(Operator *op, FNode *par1, bool maybe_negative, bool maybe_zero);
    FNode(Operator *op, FNode *par1, FNode *par2, bool maybe_negative, bool maybe_zero);
    ~FNode();
    // Methods
    std::string calculateTag();
    std::string getExpr() { return this->getOrCalculateInstruction()->expr; }
    Instruction *getOrCalculateInstruction();
    FNodeDimension &getDimension() { return dimension; }
    int getGenerationIdx() { return generation_idx; }
    bool isRoot() { return is_root; }
    nodelist_t getRoots();
    //bpy::list getRootsPython();
    bool isDimensionless() { return dimension.isDimensionless(); }
    bool notNegative() { return !maybe_negative; }
    bool notZero() { return !maybe_zero; }
    double &getValue() { return value; }
    double getPrefactor() { return prefactor; }
    double getUnitPrefactor() { return unit_prefactor; }
    double getConfidence() { return stats.q; }
    double getCovariance() { return stats.cov; }
    void setConfidence(double q_value) { stats.q = q_value; }
    void setCovariance(double cov) { stats.cov = cov; }
    void seed(double v) { value = unit_prefactor*prefactor*v; }
    std::string getOperatorTag();
    double &evaluate();
    double &evaluateRecursive();
    std::vector<FNode*> &getParents() { return parents; }
    //bpy::list getParentsPython();
    std::string calculateDimString() { return dimension.calculateString(); }
    Operator *getOperator() { return op; }
    bool containsOperator(std::string optag);
    static void registerPython();
  private:
    // Members
    int generation_idx;
    bool is_root;
    bool maybe_negative;
    bool maybe_zero;
    double unit_prefactor;
    double prefactor;
    double value;
    std::string tag;
    Operator *op;
    std::vector<FNode*> parents;
    FNodeDimension dimension;
    FNodeStats stats;
    Instruction *instruction;
};

static std::map<std::string, int> OP_PRIORITY {
    { "I", 2 },
    { "*", 1 },
    { ":", 1 },
    { "+", 0 },
    { "-", 0 },
    { "e", 2 },
    { "l", 2 },
    { "|", 2 },
    { "s", 1 },
    { "r", 1 },
    { "^", 1 },
    { "2", 1 }
};

static std::map<std::string, bool> OP_COMMUTES {
    { "I", false },
    { "*", true },
    { ":", false },
    { "+", true },
    { "-", false },
    { "e", false },
    { "l", false },
    { "|", false },
    { "s", false },
    { "r", false },
    { "2", false }
};

static std::map<std::string, bool> OP_COMMUTES_UP_TO_SIGN {
    { "I", false },
    { "*", true },
    { ":", false },
    { "+", true },
    { "-", true },
    { "e", false },
    { "l", false },
    { "|", false },
    { "s", false },
    { "r", false },
    { "2", false }
};

class Operator
{
  public:
    Operator() : tag("?") {;}
    virtual ~Operator() {;}
    // Methods
    std::string getTag() { return tag; }
    FNode *generateAndCheck(FNode *f1, FNodeCheck &chk);
    FNode *generateAndCheck(FNode *f1, FNode *f2, FNodeCheck &chk);
    virtual std::string format(std::vector<std::string> &args) { assert(false); }
    virtual double evaluate(std::vector<FNode*> &fnodes) { return -1; }
    virtual double evaluateRecursive(std::vector<FNode*> &fnodes) { return -1; }
  protected:
    virtual bool checkInput(FNode *f1) { assert(false); }
    virtual FNode* generate(FNode *f1) { assert(false); }
    virtual bool checkInput(FNode *f1, FNode *f2) { assert(false); }
    virtual FNode* generate(FNode *f1, FNode *f2) { assert(false); }
    // Members
    std::string tag;
};

class OIdent : public Operator
{
  public:
    OIdent() { tag = "I"; }
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue(); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return fnodes[0]->evaluateRecursive(); }
};

class OExp : public Operator
{
  public:
    OExp() { tag = "e"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return std::exp(fnodes[0]->getValue()); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return std::exp(fnodes[0]->evaluateRecursive()); }
};

class OLog : public Operator
{
  public:
    OLog() { tag = "l"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return std::log(fnodes[0]->getValue()); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return std::log(fnodes[0]->evaluateRecursive()); }
};

class OMod : public Operator
{
  public:
    OMod() { tag = "|"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return std::abs(fnodes[0]->getValue()); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return std::abs(fnodes[0]->evaluateRecursive()); }
};

class OSqrt : public Operator
{
  public:
    OSqrt() { tag = "s"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    double evaluate(std::vector<FNode*> &fnodes) { return std::sqrt(fnodes[0]->getValue()); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return std::sqrt(fnodes[0]->evaluateRecursive()); }
};

class OInv : public Operator
{
  public:
  OInv() { tag = "r"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    double evaluate(std::vector<FNode*> &fnodes) { return 1./fnodes[0]->getValue(); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return 1./fnodes[0]->evaluateRecursive(); }
};

class O2 : public Operator
{
  public:
    O2() { tag = "2"; }
    bool checkInput(FNode *f1);
    FNode *generate(FNode *f1);
    double evaluate(std::vector<FNode*> &fnodes) { return std::pow(fnodes[0]->getValue(), 2.0); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return std::pow(fnodes[0]->evaluateRecursive(), 2.0); }
};

class OPlus : public Operator
{
  public:
    OPlus() { tag = "+"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()+fnodes[1]->getValue(); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return fnodes[0]->evaluateRecursive()+fnodes[1]->evaluateRecursive(); }
};

class OMinus : public Operator
{
  public:
    OMinus() { tag = "-"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()-fnodes[1]->getValue(); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return fnodes[0]->evaluateRecursive()-fnodes[1]->evaluateRecursive(); }
};

class OMult : public Operator
{
  public:
    OMult() { tag = "*"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    std::string format(std::vector<std::string> &args);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()*fnodes[1]->getValue(); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return fnodes[0]->evaluateRecursive()*fnodes[1]->evaluateRecursive(); }
};

class ODiv : public Operator
{
  public:
    ODiv() { tag = ":"; }
    bool checkInput(FNode *f1, FNode *f2);
    FNode *generate(FNode *f1, FNode *f2);
    double evaluate(std::vector<FNode*> &fnodes) { return fnodes[0]->getValue()/fnodes[1]->getValue(); }
    double evaluateRecursive(std::vector<FNode*> &fnodes) { return fnodes[0]->evaluateRecursive()/fnodes[1]->evaluateRecursive(); }
};

class OP_MAP
{
  public:
    typedef std::map<std::string, Operator*> op_map_t;
    static Operator *get(std::string);
    ~OP_MAP();
  private:
    OP_MAP();
    op_map_t op_map;
    static OP_MAP *getInstance();
    static OP_MAP *instance;
};

class FGraph
{
  public:
    typedef std::vector<Operator*> op_vec_t;
    typedef std::vector<FNode*>::iterator fgraph_it_t;
    FGraph() {;}
    FGraph(std::string corr_measure, double min_exp, double max_exp, double rank_coeff);
    ~FGraph();
    // Methods
    void addRootNode(
        std::string varname, 
        std::string varplus,
        std::string varzero, 
        double unit_prefactor, 
        std::string unit);
    nodelist_t &getRoots() { return root_fnodes; }
    //bpy::list getRootsPython();
    nodelist_t &getFNodes() { return fnodes; }
    void registerNewNode(FNode *new_node);
    void generate();
    void addLayer(std::string uops, std::string bops);
    void generateLayer(op_vec_t &uops, op_vec_t &bops);
    void apply(matrix_t &input, matrix_t &output);
    void apply(double *input, double *output, int n_samples);
    void applyAndCorrelate(matrix_t &X_in, matrix_t &X_out, matrix_t &Y_in, matrix_t &cov_out);
    void applyAndCorrelate(double*, double*, double*, double*, int, int);
    void evaluateSingleNode(FNode *fnode, matrix_t &input, matrix_t &output);
    int size() { return fnodes.size(); }
    fgraph_it_t beginNodes() { return fnodes.begin(); }
    fgraph_it_t endNodes() { return fnodes.end(); }
    //bpy::object applyNumpy(bpy::object &np_input, std::string np_dtype);
    //bpy::object applyAndCorrelateNumpy(bpy::object &np_X, bpy::object &np_y, std::string np_dtype);
    void applyNumpy(
        py::array_t<double>,
        py::array_t<double>,
        int n_samples);
    void applyAndCorrelateNumpy(
        py::array_t<double>,
        py::array_t<double>,
        py::array_t<double>,
        py::array_t<double>,
        int n_samples,
        int n_targets);
    //bpy::object evaluateSingleNodeNumpy(FNode *fnode, bpy::object &np_input, std::string np_dtype);
    static void registerPython();
  private:
    // Members
    std::string correlation_measure;
    double unit_min_exp;
    double unit_max_exp;
    double rank_coeff;
    std::vector<FNode*> root_fnodes;
    std::vector<FNode*> fnodes;
    std::vector<op_vec_t> uop_layers;
    std::vector<op_vec_t> bop_layers;
    std::map<std::string, Operator*> uop_map;
    std::map<std::string, Operator*> bop_map;
    std::map<std::string, FNode*> fnode_map;
};

void zscoreMatrixByColumn(
    matrix_t &X);

void zscoreMatrixByColumn(
    double *X, 
    int n_rows, 
    int n_cols);

void mapMatrixColumnsOntoRanks(
    matrix_t &M_in, 
    matrix_t &M_out);

void correlateMatrixColumnsPearson(
    matrix_t &X_in, 
    matrix_t &Y_in, 
    matrix_t &cov_out);

void correlateMatrixColumnsPearson(
    double *X_in, 
    double *Y_in, 
    double *cov_out, 
    int n, 
    int k, 
    int l);

void correlateMatrixColumnsSpearman(
    matrix_t &X_in, 
    matrix_t &Y_in, 
    matrix_t &cov_out);

void correlateMatrixColumnsAUROC(
    matrix_t &X_in, 
    matrix_t &Y_in, 
    matrix_t &cov_out);


}}

#endif
