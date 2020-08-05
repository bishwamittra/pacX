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

(constraint (= (func true true true true false true true false true true false false 0.9610259578898896 false true false ) false))
(constraint (= (func false true true true true true true true false true true true 0.5845975596530224 true false true ) false))
(constraint (= (func false true true false true false true false false false true true 0.22238397788267006 false false true ) false))
(constraint (= (func true false true false true false true true true true false false 0.06300780564670516 false true true ) false))
(constraint (= (func false true true true false true false true true true false false 0.9883232049649384 false false false ) false))
(constraint (= (func true false false true false false false true true true true true 0.9174236955086524 false true true ) false))
(constraint (= (func true true true false false false false true false false false false 0.313260129990212 true true false ) false))
(constraint (= (func true true true true false false false true false true false false 0.9059681863338496 true true true ) false))
(constraint (= (func true false false false false false false false true true true false 0.28110143514994834 false true true ) false))
(constraint (= (func false false false false true true false true false true false false 0.8854042022415575 true true false ) false))
(constraint (= (func false true false false false true false true true true false false 0.28263908731654874 false false false ) false))
(constraint (= (func true true true true true true false false false false true true 0.6545714857019751 false false false ) false))
(constraint (= (func false false true true false true true true false true false true 0.5065543294775944 true false true ) false))
(constraint (= (func false true true false false false false true true false true false 0.6345327848982645 true false false ) false))
(constraint (= (func false false true false true true false true false true true false 0.6212272453798966 true false true ) false))
(constraint (= (func false false false false true false true true true true false false 0.08542096243007946 true true false ) false))
(constraint (= (func true true false true true false true false true true false true 0.36142559463908 false true false ) false))


(check-synth)
