(set-logic LRA)

(define-fun func ((x_0 Real) (x_1 Real) (x_2 Real) (x_3 Real)) Real (ite (= x_0 x_2) x_0 0))
(constraint (= (func (- 1.1) 1 1 42 ) 0))


(check-synth)
