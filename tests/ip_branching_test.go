package tests

import (
	"testing"

	"github.com/chriso345/gore/assert"
	"github.com/chriso345/gspl/lp"
	"github.com/chriso345/gspl/solver"
)

// linExpr builds the expression sum(coefs[i] * vars[i]); variables without a
// coefficient are left out.
func linExpr(vars []lp.LpVariable, coefs ...float64) lp.LpExpression {
	terms := make([]lp.LpTerm, len(coefs))
	for i, c := range coefs {
		terms[i] = lp.NewTerm(c, vars[i])
	}
	return lp.NewExpression(terms)
}

// Branching on a general integer must bound the variable (x <= floor or
// x >= ceil) rather than fix it, otherwise optima away from the LP relaxation
// are cut off.
func Test_IPBranchingBoundsVariable(t *testing.T) {
	x := lp.NewVariable("x", lp.LpCategoryInteger)
	y := lp.NewVariable("y", lp.LpCategoryInteger)
	vars := []lp.LpVariable{x, y}

	t.Run("optimum below the floor of the relaxation", func(t *testing.T) {
		// LP relaxation: (2.25, 3.75). IP optimum: (0, 5) with objective 40.
		prog := lp.NewLinearProgram("below floor", vars)
		prog.AddObjective(lp.LpMaximise, linExpr(vars, 5, 8))
		prog.AddConstraint(linExpr(vars, 1, 1), lp.LpConstraintLE, 6)
		prog.AddConstraint(linExpr(vars, 5, 9), lp.LpConstraintLE, 45)

		sol, err := solver.Solve(&prog)
		assert.Nil(t, err)
		assert.Equal(t, sol.Status.String(), lp.LpStatusOptimal.String())
		assert.IsClose(t, sol.ObjectiveValue, 40, 1e-5)
		assert.IsClose(t, sol.PrimalSolution.AtVec(0), 0, 1e-5)
		assert.IsClose(t, sol.PrimalSolution.AtVec(1), 5, 1e-5)
	})

	t.Run("optimum at a bound of the other variable", func(t *testing.T) {
		// LP relaxation: (3, 1.5). IP optimum: (4, 0) with objective 20.
		prog := lp.NewLinearProgram("other bound", vars)
		prog.AddObjective(lp.LpMaximise, linExpr(vars, 5, 4))
		prog.AddConstraint(linExpr(vars, 6, 4), lp.LpConstraintLE, 24)
		prog.AddConstraint(linExpr(vars, 1, 2), lp.LpConstraintLE, 6)

		sol, err := solver.Solve(&prog)
		assert.Nil(t, err)
		assert.Equal(t, sol.Status.String(), lp.LpStatusOptimal.String())
		assert.IsClose(t, sol.ObjectiveValue, 20, 1e-5)
		assert.IsClose(t, sol.PrimalSolution.AtVec(0), 4, 1e-5)
		assert.IsClose(t, sol.PrimalSolution.AtVec(1), 0, 1e-5)
	})

	t.Run("up branch", func(t *testing.T) {
		// LP relaxation: x = 2.5. IP optimum: x = 3, reached via x >= 3.
		prog := lp.NewLinearProgram("up branch", vars)
		prog.AddObjective(lp.LpMinimise, linExpr(vars, 1, 0))
		prog.AddConstraint(linExpr(vars, 1, 0), lp.LpConstraintGE, 2.5)

		sol, err := solver.Solve(&prog)
		assert.Nil(t, err)
		assert.Equal(t, sol.Status.String(), lp.LpStatusOptimal.String())
		assert.IsClose(t, sol.ObjectiveValue, 3, 1e-5)
		assert.IsClose(t, sol.PrimalSolution.AtVec(0), 3, 1e-5)
	})
}
