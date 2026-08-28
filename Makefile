
test:
	go test

exp-coefs:
	EXP_COEFS=1 go test

log-coefs:
	LOG_COEFS=1 go test

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
