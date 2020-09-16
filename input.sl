(set-logic LRA)

(define-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool)) Bool (and hair milk toothed backbone breathes tail (not feathers) (not eggs) (not aquatic) (not predator) (not venomous) (not fins) (not domestic) (or (and airborne (= legs (/ 1 4)) (not catsize)) (and catsize (not airborne) (= legs (/ 1 2))))))
(constraint (= (func false false true false false true true true true false false true 0.0 true false true ) false))


(check-synth)
