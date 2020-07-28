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
                            (- R)
                        )
                    )
                )
            

)

(constraint (= (func true false true true false false true true true true false false 1 true false false ) false))
(constraint (= (func false false false true true false false true true false false true 0 false true false ) true))
(constraint (= (func true false false true true false true false true true false true 1 true false true ) false))
(constraint (= (func false false false true true true false false true false true true 1 false false false ) true))
(constraint (= (func false false true false true false true false false true false false 0 true true true ) false))
(constraint (= (func false true false true false true true true true false false true 1 false true true ) true))
(constraint (= (func true true true false true false false false false false false true 1 true true false ) false))
(constraint (= (func true false true false false true true false false true false true 0 false true true ) false))
(constraint (= (func true true false true false true true true true false false false 1 false false true ) true))
(constraint (= (func false true false true true false true true true false false false 0 true false false ) false))
(constraint (= (func false true false true false true true false false false true true 1 false false false ) false))
(constraint (= (func true true true true false true false true true true false false 0 false false false ) false))


(check-synth)
