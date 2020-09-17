(set-logic LRA)

(synth-fun func ((hair Bool) (feathers Bool) (eggs Bool) (milk Bool) (airborne Bool) (aquatic Bool) (predator Bool) (toothed Bool) (backbone Bool) (breathes Bool) (venomous Bool) (fins Bool) (legs Real) (tail Bool) (domestic Bool) (catsize Bool) ) Bool


                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) (Var_Bool Bool) (Var_Real Real) (Const_Real Real))

                ;; Define the grammar for allowed implementations
                (
                    (
                        Formula Bool (
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
                            (Constant Bool)
                            Var_Bool
                            (not Var_Bool)
                            (> Var_Real Const_Real)
                            (< Var_Real Const_Real)
                            )
                    )
                    (
                        Var_Bool Bool (
                            (Variable Bool)
                        )
                    )
                    (
                        Var_Real Real (
                            (Variable Real)
                        )
                    )
                    (
                        Const_Real Real (
                            0.0 0.25 0.5 0.75 1.0
                        )
                    )
                    
                )

            

)

(constraint (= (func false false false false false false false true false true false true 0.329 false false false ) false))
(constraint (= (func true true false false true true true true true true true false 0.699 false true true ) false))
(constraint (= (func true false false false true false false true false true false true 0.626 true false true ) false))
(constraint (= (func true true false false false false true false false true false true 0.064 true false false ) false))
(constraint (= (func true true false false false true false false false true false true 0.093 true true false ) false))
(constraint (= (func false false true true true true false false false false false true 0.612 true true false ) true))
(constraint (= (func true true true true false true true false true true true false 0.384 false false false ) false))
(constraint (= (func false true false true false true false true false false false false 0.866 true false true ) false))
(constraint (= (func false false false true true false false true false false false true 0.884 false false true ) true))
(constraint (= (func false false true false true false true true true false true true 0.397 false true true ) false))
(constraint (= (func true true false true true false false false false false false true 0.145 false false true ) true))
(constraint (= (func false true false true true false true false true true true false 0.533 true false true ) false))
(constraint (= (func false false true true true true true true true true false true 0.914 false true false ) false))
(constraint (= (func true false true true false true true false false false false true 0.097 false false true ) true))
(constraint (= (func false true false true true false false true false true false true 0.596 false true true ) false))


(check-synth)
