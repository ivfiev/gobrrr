package gobrrr

import (
	"fmt"
	"math"
	"os"
	"testing"
)

func explicitTest(t *testing.T, envVar string) {
	if os.Getenv(envVar) != "1" {
		t.Skip()
	}
}

func assertRelEqual(t *testing.T, want, got []float64, eps float64) {
	for i := range len(got) {
		err := math.Abs(got[i] - want[i])
		if want[i] > eps {
			err /= want[i]
		}
		if err > eps {
			t.Fatalf("%.12f != %.12f\n", got[i], want[i])
		}
	}
}

func calcError(want, got any) (float64, float64, float64) {
	const eps = 1e-9
	switch want.(type) {
	case float64:
		want, got := want.(float64), got.(float64)
		return math.Abs(want-got) / want, want, got
	case []float64:
		want, got := want.([]float64), got.([]float64)
		err, errWant, errGot := 0.0, 0.0, 0.0
		for i := range want {
			w := float64(want[i])
			g := float64(got[i])
			rel := math.Abs(w-g) / w
			if rel > err {
				err = rel
				errWant = w
				errGot = g
			}
		}
		return err, errWant, errGot
	case [][]float64:
		want, got := want.([][]float64), got.([][]float64)
		err, errWant, errGot := 0.0, 0.0, 0.0
		for i := range want {
			errT, errW, errG := calcError(want[i], got[i])
			if errT > err {
				err = errT
				errWant = errW
				errGot = errG
			}
		}
		return err, errWant, errGot
	case int:
		want, got := want.(int), got.(int)
		return math.Abs(float64(want - got)), float64(want), float64(got)
	default:
		return -1.0, -1.0, -1.0
	}
}

func printError(label string, want, got any) {
	total, want, got := calcError(want, got)
	fmt.Printf("%s: rel err %.12f, want %.12f, got %.12f\n", label, total, want, got)
}
