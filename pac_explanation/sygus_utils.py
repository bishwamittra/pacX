import regex
import numpy as np

def push(obj, l, depth):
        while depth:
            l = l[-1]
            depth -= 1
        l.append(obj)

def parse_parentheses(s):
        groups = []
        depth = 0
        blocks = []

        try:
            block = ""
            for i in range(len(s)):
                char = s[i]
                if char == " ":
                    continue
                if char == '(':
                    push([], groups, depth)
                    depth += 1
                elif char == ')':
                    depth -= 1
                else:
                    block += char
                    if(s[i+1] in ['(',')',' ']):
                        push(block, groups, depth)
                        if(block not in blocks):
                            blocks.append(block)
                        block = ""
        except IndexError:
            print(s)
            raise ValueError('Parentheses mismatch')

        if depth > 0:
            # print(s)
            raise ValueError('Parentheses mismatch')
        else:
            return groups[0], blocks 
        

def eval(exp):
    m = regex.match(r'\(([-+\/\*]) ((?R)) ((?R))\)|([0-9]+\.?[0-9]+?)', exp)
    if m.group(1) and m.group(2) and m.group(3):  # exp is a procedure call
        return eval('%s %s %s' % (eval(m.group(2)),
                                m.group(1),
                                eval(m.group(3))))
    if m.group(4):  # exp is a number
        return str(eval(m.group(4)))


def add_context_free_grammar(rule_bound_k, rule_type, _feature_data_type, _real_attribute_domain_info, _categorical_attribute_domain_info, syntactic_grammar):

        
        dic_operator = {
            "CNF" : {
                "outer" : "and",
                "inner" : "or"
            },
            "DNF" : {
                "outer" : "or",
                "inner" : "and"
            }

        }


        clause_constraints = None
        if(rule_bound_k == -1):
            clause_constraints = "B (" + dic_operator[rule_type]["inner"] + " B Clause)"
        elif(rule_bound_k == 1):
            clause_constraints = "B"
        else:
            clause_constraints = "(" + dic_operator[rule_type]["inner"] + " " + (" ").join(["B" for _ in range(rule_bound_k)]) + ")" 
        
        # print(clause_constraints)

        bool_features_str = (" ").join([ _feature + " (not " + _feature + ")" for _feature in list(_feature_data_type.keys()) if _feature_data_type[_feature] == "Bool"])
        real_feature_header_str = (" ").join(["(Const_" + _feature + " Real)" for _feature in _real_attribute_domain_info])
        categorical_feature_header_str = (" ").join(["(Const_" + _feature + " Real)" for _feature in _categorical_attribute_domain_info])
        real_feature_syntactic_constraints = (" ").join(["(> " + _feature + " Const_" + _feature + ") (< " + _feature + " Const_" + _feature + ")" for _feature in _real_attribute_domain_info])
        categorical_feature_syntactic_constraints = (" ").join(["(= " + _feature + " Const_" + _feature + ") (not (= " + _feature + " Const_" + _feature + "))" for _feature in _categorical_attribute_domain_info])        
        bins = 6 
        real_feature_constant_str = ("\n").join(
                ["(Const_" + _feature + " Real (" 
                    + (" ").join(map(str, list(np.linspace(_real_attribute_domain_info[_feature][1], _real_attribute_domain_info[_feature][0], bins))))
                    + "))" for _feature in _real_attribute_domain_info])
        categorical_feature_constant_str = ("\n").join(
                ["(Const_" + _feature + " Real (" 
                    + (" ").join(map(str, _categorical_attribute_domain_info[_feature]))
                    + "))" for _feature in _categorical_attribute_domain_info])

        
        # print(real_feature_header_str)
        # print(real_feature_syntactic_constraints) 
        # print(real_feature_constant_str)

        s = ""
        
        # Bool and either of Real and categorical or both
        if("Bool" in list(_feature_data_type.values()) and ("Real" in list(_feature_data_type.values()) or "Categorical" in list(_feature_data_type.values()))):
            s = """
                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) (Var_Bool Bool) {} {})

                ;; Define the grammar for allowed implementations
                (
                    (
                        Formula Bool (
                            Clause
                            ({} Clause Formula)
                        )
                    )
                    (
                        Clause Bool (
                            {}
                        )
                    )
                    (
                        B Bool (
                            (Constant Bool)
                            Var_Bool
                            {}
                            {}
                            )
                    )
                    (
                        Var_Bool Bool (
                            {}
                        )
                    )
                    {}
                    {}
                    
                )

            """.format(real_feature_header_str, categorical_feature_header_str, dic_operator[rule_type]["outer"], clause_constraints, real_feature_syntactic_constraints, categorical_feature_syntactic_constraints, bool_features_str, real_feature_constant_str, categorical_feature_constant_str)
        elif("Real" in list(_feature_data_type.values()) or "Categorical" in list(_feature_data_type.values())):
            s = """
                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool) {} {})

                ;; Define the grammar for allowed implementations
                (
                    (
                        Formula Bool (
                            Clause
                            ({} Clause Formula)
                        )
                    )
                    (
                        Clause Bool (
                            {}
                        )
                    )
                    (
                        B Bool (
                            (Constant Bool)
                            {}
                            {}
                            )
                    )
                    {}
                    {}
                    
                )

            """.format(real_feature_header_str, categorical_feature_header_str, dic_operator[rule_type]["outer"], clause_constraints, real_feature_syntactic_constraints, categorical_feature_syntactic_constraints, real_feature_constant_str, categorical_feature_constant_str)
        
        # only Bool
        elif("Bool" in list(_feature_data_type.values())):
            s = """
                ;; Declare the non-terminals that would be used in the grammar
                ((Formula Bool) (Clause Bool) (B Bool))

                ;; Define the grammar for allowed implementations
                (
                    (
                        Formula Bool (
                            Clause
                            ({} Clause Formula)
                        )
                    )
                    (
                        Clause Bool (
                            {}
                        )
                    )
                    (
                        B Bool (
                            {}
                            (Constant Bool)
                            )
                    )
                    
                )
             """.format(dic_operator[rule_type]["outer"], clause_constraints, bool_features_str)
        
        else:
            raise ValueError("Syntactic constraint cannot be constructed")

        
        # no syntactic costraints added
        if(not syntactic_grammar):
            s = ""
        return s + "\n\n"
    

        
