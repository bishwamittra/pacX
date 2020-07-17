(set-logic LRA)

(synth-fun func ((x_0 Real) (x_1 Real) (x_2 Real) (x_3 Real) ) Real


            (( y_term Real ) ( y_cons Real ) ( y_pred Bool ))
            (( y_term Real ( y_cons
                ( Variable Real )
                (- y_term )
                (+ y_term y_term )
                (- y_term y_term )
                (* y_cons y_term )
                (* y_term y_cons )
                ( ite y_pred y_term y_term )))
            ( y_cons Real (( Constant Real )))
            ( y_pred Bool ((= y_term y_term )
                ( > y_term y_term )
                ( >= y_term y_term )
                ( < y_term y_term )
                ( <= y_term y_term ))))
                    

)

(constraint (= (func 1 2 1 1 ) 1))
(constraint (= (func (- 1.1) 1 1 42 ) 0))


(check-synth)
