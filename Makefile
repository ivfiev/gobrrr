
test:
	go test

exp-coefs-spsa:
	EXP_COEFS_SPSA=1 go test

log-coefs-spsa:
	LOG_COEFS_SPSA=1 go test

icdf-coefs-spsa:
	ICDF_COEFS_SPSA=1 go test

bench:
	go test -bench=.

bench-spsa:
	go test -bench=SPSA

bench-exp:
	go test -bench=Exp

bench-log:
	go test -bench=Log

bench-mm:
	go test -bench=MM -benchmem

bench-rng:
	go test -bench=RNG
