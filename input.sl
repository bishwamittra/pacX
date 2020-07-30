(set-logic LRA)

(define-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool)) Bool milk)
(constraint (= (func false true true false true false false false true true false false 0.25 true false false ) false))


(check-synth)
