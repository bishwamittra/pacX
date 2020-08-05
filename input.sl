(set-logic LRA)

(synth-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool) ) Bool


                ;; Declare the non-terminals that would be used in the grammar
                ((B Bool) (C Real) (R Real))

                ;; Define the grammar for allowed implementations
                (
                    (
                        B Bool (
                            true false
                            (Variable Bool)
                            (not B)
                            (or B B)
                            (and B B)
                            (> R C)
                            (< R C)
                        )
                    )
                    (
                        C Real (
                            (Constant Real)
                        )
                    )
                    (
                        R Real (
                            (Variable Real)
                        )
                    )
                )
            

)

(constraint (= (func false true true false false true true true true true false false 0.835 true true false ) false))
(constraint (= (func true false false false true false false true false false false false 0.01 true false false ) true))
(constraint (= (func true false true false true false true true false false true true 0.995 false false false ) false))
(constraint (= (func true true true true true false true true true true true true 0.5 false false false ) true))
(constraint (= (func true false false true true false true false false true false false 0.555 true true false ) false))
(constraint (= (func false false false false true false false true true false false false 0.966 false true true ) true))
(constraint (= (func false true false false true false true false false true false true 0.507 false true true ) false))
(constraint (= (func true true false true false false false true false false false true 0.275 false false true ) true))
(constraint (= (func false false false false false true false true true false false false 0.403 false true false ) false))
(constraint (= (func true false false true true false true true false false true true 0.846 false true false ) true))
(constraint (= (func true true true true false false false true false false false false 0.101 false false false ) false))
(constraint (= (func true true false true false false true true true false true true 0.802 true false false ) true))
(constraint (= (func true false true true true true true true false false false true 0.851 false false true ) false))
(constraint (= (func true false true false true false false true true false true true 0.762 true false false ) false))


(check-synth)
