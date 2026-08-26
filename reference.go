package gobrrr

import "math/rand"

func rademacher(x []float64) {
	for i := range x {
		if rand.Float64() > 0.5 {
			x[i] = 1
		} else {
			x[i] = -1
		}
	}
}
