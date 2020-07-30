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

(constraint (= (func false true false true true true true false false true false true 0.4399834457131677 true false true ) false))
(constraint (= (func true true false true false false false true true true false false 0.5669723856750564 true true true ) true))
(constraint (= (func true false false true true true false true false true false false 0.6845565169688158 true false false ) false))
(constraint (= (func true false false false false true false false true false false false 0.28070347128916984 false true false ) false))
(constraint (= (func false true true true true false true false true true false false 0.11071380305107337 false false false ) false))
(constraint (= (func true false true true false false true true true true true false 0.35914741261007177 true false true ) true))
(constraint (= (func true true false false true true false false false true false false 0.39540466420771825 true false true ) false))
(constraint (= (func true false false false false false false false true true false false 0.29446936115780753 true false false ) true))
(constraint (= (func true false true true false false true false true false false true 0.031097497097383853 true false true ) false))
(constraint (= (func false true false false false false false false true true true false 0.8659426960416887 true true true ) true))
(constraint (= (func false false false false true true false false true true false true 0.396825361250688 true false false ) false))
(constraint (= (func true true true false true false false true true true false true 0.2307599012717847 true true false ) true))
(constraint (= (func false true false true false true true false true true true false 0.6962983171100023 false true false ) false))
(constraint (= (func false true true false true false true false true true false true 0.2958614283675828 true false false ) true))
(constraint (= (func false false false false true false false true false true false false 0.305792301405712 true false false ) false))
(constraint (= (func false false false true false false false false true true false true 0.02806393307545585 true true true ) true))


(check-synth)
