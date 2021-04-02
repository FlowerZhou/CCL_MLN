/*cppimport

*/
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#define BOOST_SPIRIT_X3_DEBUG

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/home/x3.hpp>

#include <boost/spirit/home/x3/support/ast/position_tagged.hpp>
#include <boost/spirit/home/x3/support/utility/error_reporting.hpp>
#include <boost/spirit/home/x3/support/utility/annotate_on_success.hpp>


#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>

#include <iostream>
#include <string>
#include <complex>
#include <variant>


namespace mln_logic{
    namespace x3 = boost::spirit::x3;
    struct PredicateArgument : x3::position_tagged {
        std::string type;
        std::optional<std::string> name;
        std::optional<bool> unique;
    };

    struct Predicate : x3::position_tagged {
        std::optional<char> cwa;
        std::string name;
        std::vector<PredicateArgument> args;
    };

    struct Term : x3::position_tagged {
        std::variant<std::string, int, float, double> value;
    };

    struct Atom : x3::position_tagged {
        std::string name;
        std::vector<Term> args;
    };

    struct Literal : x3::position_tagged {
        std::optional<char> tag;
        Atom atom;
    };

    struct Clause: x3::position_tagged {

        std::optional<std::vector<Literal>> antecedent;
        std::vector<Literal> consequent;

    };


    struct Rule: x3::position_tagged {

        std::variant<double, std::string> weight;
        Clause clause;

    };

    struct Program: x3::position_tagged {
        std::vector<Predicate> predicates;
        std::vector<Rule> rules;
    };

    using boost::fusion::operator<<;
}

BOOST_FUSION_ADAPT_STRUCT(
    mln_logic::PredicateArgument,
    type, name, unique
)

BOOST_FUSION_ADAPT_STRUCT(
    mln_logic::Predicate,
    cwa, name, args
)

BOOST_FUSION_ADAPT_STRUCT(
    mln_logic::Term,
    value)

BOOST_FUSION_ADAPT_STRUCT(
    mln_logic::Atom,
    name, args)

BOOST_FUSION_ADAPT_STRUCT(
    mln_logic::Literal,
    tag, atom
)

BOOST_FUSION_ADAPT_STRUCT(
    mln_logic::Clause,
    antecedent, consequent
)

BOOST_FUSION_ADAPT_STRUCT( mln_logic::Rule,
    weight, clause
)

BOOST_FUSION_ADAPT_STRUCT( mln_logic::Program,
    predicates , rules
)



namespace {
    template <typename T>
    struct as_type {
        template <typename Expr>
        auto operator[](Expr&& expr) const {
            return boost::spirit::x3::rule<struct _, T>{"as"} = boost::spirit::x3::as_parser(std::forward<Expr>(expr));
        }

        template <typename Expr>
        auto operator()(Expr&& p) const{
            return boost::spirit::x3::rule<struct tag, T> {"as"} = p;
        }
    };

    template <typename T> static const as_type<T> as = {};
}

namespace mln_parser{

    namespace x3 = boost::spirit::x3;
    namespace ascii = boost::spirit::x3::ascii;

    using x3::int_;
    using x3::lit;
    using x3::double_;
    using x3::float_;
    using x3::eoi;
    using x3::eol;
    using x3::lexeme;
    using ascii::char_;
    using x3::alnum;

    struct error_handler
    {
        template <typename Iterator, typename Exception, typename Context>
        x3::error_handler_result on_error(
            Iterator& first, Iterator const& last
          , Exception const& x, Context const& context)
        {
            auto& error_handler = x3::get<x3::error_handler_tag>(context).get();
            std::string message = "Error! Expecting: " + x.which() + " here:";
            error_handler(x.where(), message);
            return x3::error_handler_result::fail;
        }
    };

    struct ID_class ;
    struct predicate_arg_class ;
    struct predicate_class ;

    struct term_class ;
    struct atom_class ;
    struct literal_class ;
    struct clause_class ;
    struct fixed_soft_rule_class;
    struct learnable_soft_rule_class;
    struct hard_rule_class;
    struct rule_class ;
    struct program_class ;


    x3::rule<ID_class, std::string> const ID = "id";
    x3::rule<predicate_arg_class, mln_logic::PredicateArgument> const predicate_arg =  "predicate argument";
    x3::rule<predicate_class, mln_logic::Predicate> const predicate = "predicate";
    x3::rule<term_class, mln_logic::Term> const term = "term";
    x3::rule<atom_class, mln_logic::Atom> const atom = "atom";
    x3::rule<literal_class, mln_logic::Literal> const literal = "literal";
    x3::rule<clause_class, mln_logic::Clause> const clause = "clause";
    x3::rule<fixed_soft_rule_class, mln_logic::Rule> const fixed_soft_rule = "fixed_soft_rule";
    x3::rule<learnable_soft_rule_class, mln_logic::Rule> const learnable_soft_rule = "learnable_soft_rule";
    x3::rule<hard_rule_class, mln_logic::Rule> const hard_rule = "hard_rule";
    x3::rule<rule_class, mln_logic::Rule> const rule = "rule";
    x3::rule<program_class, mln_logic::Program> const program = "program";

    auto const ID_def = as<std::string>(x3::no_case[(char_('a', 'z') | '_') > * ( alnum | '_')]);
    auto const predicate_arg_def = ID > (- ID) > (- char_('!'));
    auto const predicate_def = (- char_('*')) > ID >  '(' > predicate_arg % ',' > ')';

    auto const term_def = ID | int_ | float_ | double_;

    auto const atom_def = ID > '(' > term % ',' > ')';

    auto const literal_def = - ( char_('+') | char_('!')) > atom;

    auto const clause_def = - (as<std::vector<mln_logic::Literal>>[literal % ','] >
                              "=>" ) > as<std::vector<mln_logic::Literal>>[literal % 'v'];


    auto const fixed_soft_rule_def = (double_ > clause);
    auto const learnable_soft_rule_def = ( ID  > ':' > clause);
    auto const hard_rule_def = (as<std::string>(x3::string(".")) >  clause);
    auto const rule_def = (fixed_soft_rule | learnable_soft_rule | hard_rule);


    auto const comment = x3::omit[x3::string("//") > * (char_ - eol) ] ;
//    auto const blanks = x3::omit[* x3::blank] ;

    auto const skipper = x3::blank | comment;// | blank_line;

    auto const program_def =  - predicate % eol >
//                             - rule % eol >
                             eoi;

    BOOST_SPIRIT_DEFINE(ID, predicate_arg, predicate, term, atom, literal, clause,
                fixed_soft_rule, learnable_soft_rule, hard_rule,
                rule, program);

//    BOOST_SPIRIT_X3_DEBUG((ID)(predicate_arg)(predicate)(term)(atom)(literal)(clause)
//                (fixed_soft_rule)(learnable_soft_rule)(hard_rule)
//                (rule)(program));

//    auto const existance = 'EXIST' > (ID % ',');


//
//    auto const math_comparision = math_expression > ('='|'<>'|'<'|'<='|'>'|'>='|'!=') > math_expression;
//
//    auto const math_expression = math_term > * ( ('+'|'-'|'%') > math_term);
//
//    auto const math_term = math_factor > * ( ('*'|'/'|'&'|'|'|'^'|'<<'|'>') > math_factor);
//
//    auto const math_factor = (func_expression | atomic_expression |
//                ("(" > math_expression > ")") | ('~' > math_factor) ) > - '!';
//
//    auto const clause = - existance >
//                    literal > - ( ',' >  (literal | math_comparision ) % ',') >
//                    - ( ',' >  '[' > bool_expression > ']' ) >
//                    "=>" >
//                    literal > - ( 'v' >  (literal | math_comparision ) % ',') >
//                    - ( 'v' >  '[' > bool_expression > ']' );


    struct ID_class : error_handler, x3::annotate_on_success {};
    struct predicate_arg_class : error_handler, x3::annotate_on_success {};
    struct predicate_class : error_handler, x3::annotate_on_success {};

    struct term_class : error_handler, x3::annotate_on_success {};
    struct atom_class : error_handler, x3::annotate_on_success {};
    struct literal_class : error_handler, x3::annotate_on_success {};
    struct clause_class : error_handler, x3::annotate_on_success {};
    struct rule_class : error_handler, x3::annotate_on_success {};
    struct program_class : error_handler, x3::annotate_on_success {};

    bool parse_mln_program(const std::string & input)
    {

        using boost::spirit::x3::phrase_parse;
        using boost::spirit::x3::ascii::space;

        mln_logic::Program program_instance;

        auto first = input.begin();
        auto last = input.end();


        typedef std::string::const_iterator iterator_type;

        using boost::spirit::x3::with;
        using boost::spirit::x3::error_handler_tag;
        using error_handler_type = boost::spirit::x3::error_handler<iterator_type>;

        // Our error handler
        error_handler_type error_handler(first, last, std::cerr);

        // Our parser

        auto const parser =
            // we pass our error handler to the parser so we can access
            // it later in our on_error and on_sucess handlers
            with<error_handler_tag>(std::ref(error_handler))
            [
                - rule % eol
            ];
        std::vector<mln_logic::Predicate> predicates;
        std::vector<mln_logic::Rule> rules;

        bool r = phrase_parse(first, last, parser, skipper, rules);
        std::cout << "num of predicate : " << program_instance.predicates.size() << std::endl;
        for (auto pred: program_instance.predicates)
            std::cout << "name :" << pred.name << std::endl;
        std::cout << "num of rules : " << rules.size() << std::endl;

        if (first != last){
            return false;
        }
        return r;

    }
}

PYBIND11_MODULE(parser_impl, m) {
    m.def("parse_mln_program", &mln_parser::parse_mln_program,
        py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>());
}

