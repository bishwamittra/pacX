(set-logic LRA)

(synth-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool) ) Bool


                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) (C Real) (R Real))

                ;; Define the grammar for allowed implementations
                (
                    (
                        Formula Bool (
                            true
                            Clause
                            (or Clause Formula)
                        )
                    )
                    (
                        Clause Bool (
                            B
                            (and B Clause)
                        )
                    )
                    (
                        B Bool (
                            (Variable Bool)
                            (not B)
                            (= R C)
                            (> R C)
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

(constraint (= (func false false true false true false false false true false true false 0.107 true true false ) false))
(constraint (= (func true false false true true false false false true false false false 0.591 true true false ) true))
(constraint (= (func true true true true false true false true true false true false 0.364 false false false ) false))


(check-synth)
