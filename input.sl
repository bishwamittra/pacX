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
                            0 0.25 0.5 0.75 1
                        )
                    )
                    
                )

            

)

(constraint (= (func true false false true true false false true true true false false 0.25 true false false ) true))
(constraint (= (func false true true false false false false false false true true true 0.476 false false true ) false))
(constraint (= (func true false true true true true true true false false false false 0.448 false false false ) false))
(constraint (= (func false true true false true true true false true true false true 0.504 false true true ) false))
(constraint (= (func false true false false false true true false true true false false 0.97 false true true ) false))
(constraint (= (func false false false true false false true false true false true false 0.226 true true false ) false))
(constraint (= (func true false false true true false true false true false false true 0.825 true true false ) false))
(constraint (= (func true false true false true true false true false false true true 0.127 true false false ) false))
(constraint (= (func true false true true true true false false true true true false 0.247 false true true ) false))
(constraint (= (func true false false true true false false true true false true false 0.54 true false false ) false))
(constraint (= (func true false true false true false false false true true false false 0.954 false true true ) false))
(constraint (= (func true true true true false false true true false true true false 0.683 false false true ) false))
(constraint (= (func true false true false true true false false true true true true 0.612 true true true ) false))
(constraint (= (func false false false true true false false true false true false true 0.459 true false false ) false))
(constraint (= (func true false false false true false false false false true true true 0.482 true false false ) false))
(constraint (= (func true true true false true false true true true true false true 0.305 false false false ) false))
(constraint (= (func true false false false false true true false false true true false 0.564 true false false ) false))
(constraint (= (func false true true false true false true false false false false false 0.444 true true true ) false))
(constraint (= (func true false true true true true false true false true true true 0.718 true true true ) false))


(check-synth)
