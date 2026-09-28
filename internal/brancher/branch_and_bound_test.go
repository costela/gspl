package brancher

import (
	"math"
	"sync/atomic"
	"testing"

	"github.com/chriso345/gore/assert"
	"github.com/chriso345/gspl/internal/common"
	"github.com/chriso345/gspl/internal/errors"
	"gonum.org/v1/gonum/mat"
)

// minimal SCF for testing
func newTestSCF(primal []float64) *common.StandardComputationalForm {
	objVal := 0.0
	status := common.SolverStatusOptimal
	vec := mat.NewVecDense(len(primal), primal)
	return &common.StandardComputationalForm{
		PrimalSolution: vec,
		ObjectiveValue: &objVal,
		Status:         &status,
		NumPrimals:     len(primal),
		SlackIndices:   make([]int, len(primal)),
	}
}

// newFractionalRootSCF is min -x - y s.t. 2x + 2y + s = 5 with x, y integer.
// Its relaxation and the x <= 2 child are both fractional, so solving it
// branches at least twice.
func newFractionalRootSCF() *common.StandardComputationalForm {
	objVal := 0.0
	status := common.SolverStatusNotSolved
	return &common.StandardComputationalForm{
		Objective:      mat.NewVecDense(3, []float64{-1, -1, 0}),
		Constraints:    mat.NewDense(1, 3, []float64{2, 2, 1}),
		RHS:            mat.NewVecDense(1, []float64{5}),
		PrimalSolution: mat.NewVecDense(3, nil),
		ObjectiveValue: &objVal,
		Status:         &status,
		SlackIndices:   []int{-1, -1, 2},
		NumPrimals:     2,
		VarCategories:  []common.VarCategory{common.VarCategoryInteger, common.VarCategoryInteger, common.VarCategoryContinuous},
	}
}

// An error below the root must fail the solve rather than silently drop the subtree.
func TestBranchAndBound_PropagatesChildError(t *testing.T) {
	ip := &common.IntegerProgram{SCF: newFractionalRootSCF(), BestObj: math.Inf(1)}
	var calls atomic.Int32
	ip.Branch = func(node *common.Node) ([]*common.Node, error) {
		if calls.Add(1) > 1 {
			return nil, errors.New(errors.ErrUnknown, "injected branching failure", nil)
		}
		return DefaultBranch(node)
	}

	err := BranchAndBound(ip, common.DefaultSolverConfig())
	assert.True(t, calls.Load() > 1)
	assert.NotNil(t, err)
}
