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

(constraint (= (func true false false false false false true false true true true true 0.352 false true true ) false))
(constraint (= (func false false true true false true true true true true false true 0.477 true true false ) false))
(constraint (= (func false false false true false false false false false false false false 0.284 false false false ) false))
(constraint (= (func true false true false false false false false false true true true 0.292 false false true ) false))
(constraint (= (func true false false false false false false false true true false true 0.668 true true true ) false))
(constraint (= (func true false false true true true false true true false false false 0.732 true false true ) false))
(constraint (= (func false false true false false true true true true false false false 0.779 false true true ) false))
(constraint (= (func true true false true true false true true false true false false 0.345 false true true ) false))
(constraint (= (func true false true true true false true false false true false true 0.382 false false false ) false))
(constraint (= (func true false false false false true true false false true false false 0.998 false false true ) false))
(constraint (= (func true false false true true false false false false true true true 0.638 false false true ) false))
(constraint (= (func false false false true false false false true true false true false 0.005 true false true ) false))
(constraint (= (func true true false false false true true true false true false false 0.968 false true true ) false))
(constraint (= (func true false false true false false false false true true false false 0.328 false true false ) false))
(constraint (= (func true true true false false true true true false true false true 0.642 true false true ) false))
(constraint (= (func true false false true true true true false false true true false 0.48 true true true ) false))
(constraint (= (func true true false true false true false false true true false true 0.761 true true false ) false))
(constraint (= (func false true false true true false false true true true true true 0.4 true true false ) false))
(constraint (= (func true false false false true false true false true true true false 0.749 true true true ) false))
(constraint (= (func true true false false false true true false true false false false 0.1 true true true ) false))
(constraint (= (func false true true false true true true false true false true false 0.579 true true false ) false))
(constraint (= (func false false true true true false true true false false false false 0.019 true false true ) false))
(constraint (= (func false false false false true false false false false true true true 0.475 true true false ) false))
(constraint (= (func false true true true false true true false true true true false 0.089 true false true ) false))
(constraint (= (func false true false true false true true true false false true false 0.43 true false true ) false))
(constraint (= (func true false false true false true false false true true true true 0.563 true false false ) false))
(constraint (= (func true false true false true true true false true true true true 0.633 true true true ) false))
(constraint (= (func true true false false true true false false false true true false 0.544 true true false ) false))
(constraint (= (func true false false false false true true true false false false false 0.452 true false false ) false))
(constraint (= (func false true true false false true true false false true false true 0.203 true true true ) false))
(constraint (= (func true false true true false true false false true true true false 0.658 true true true ) false))
(constraint (= (func false true false true false false false false true true false true 0.211 true true false ) false))
(constraint (= (func false true false false false false false true true true false false 0.83 false false true ) false))
(constraint (= (func true true false false false true false true true true false false 0.314 false false true ) false))
(constraint (= (func false false true false false false true true true true true false 0.845 true true false ) false))
(constraint (= (func true false false true true false false true true true true true 0.026 false true false ) false))
(constraint (= (func false true true true true false true false false true false true 0.01 false true false ) false))
(constraint (= (func false false false false true false true false false true true true 0.517 true false true ) false))
(constraint (= (func true false true false false true true false false false false false 0.872 true true false ) false))
(constraint (= (func true false false true false false false true true true false true 0.949 false true true ) false))
(constraint (= (func false true true true true true false true false true true false 0.43 true true true ) false))
(constraint (= (func false false false false true false false false true true true false 0.098 false false false ) false))
(constraint (= (func true false true false false true true false false true false true 0.157 true false true ) false))
(constraint (= (func false false true true true false true false true false false false 0.91 true false false ) false))
(constraint (= (func false true false true true false false true true true false false 0.931 true false false ) false))
(constraint (= (func true true false true true true false true true true true false 0.959 false false false ) false))
(constraint (= (func false true false true false false true false false true false true 0.165 true true false ) false))
(constraint (= (func false true true true false true true false false false true true 0.153 true true true ) true))


(check-synth)
