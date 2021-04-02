grammar MLN;


WS  :   ( ' '
        | '\t'
        | '\r'
        | '\n'
        ) 
    -> channel(HIDDEN);

COMMENT
    :  ( '//' ~('\n'|'\r')* '\r'? '\n'
    |   '/*' .*? '*/')
    -> channel(HIDDEN) ;

NOT 	:	'!';
PLUS   : '+';
MINUS   : '-';
ASTERISK:	'*';
PERIOD: '.';
EXIST	:	'EXIST' | 'Exist' | 'exist';
IMPLIES : '=>';

STRING :
   '"'
   ( ESC  | ~('"'|'\\'|'\n'|'\r') )*
   '"'
    ;

fragment
ESC
    :   '\\'
        (       'n'
        |       'r'
        |       't'
        |       'b'
        |       'f'
        |       '"'
        |       '\''
        |       '/'
        |       '\\'
        | 'u' HEXDIGIT HEXDIGIT HEXDIGIT HEXDIGIT
        | ~('u'|'r'|'n'|'t'|'b'|'f'|'"'|'\''|'/'|'\\')
        )
    ;

fragment
HEXDIGIT
  : '0'..'9' | 'A'..'F' | 'a'..'f'
  ;

NUMBER :  INTEGER | FLOAT;
fragment
INTEGER : '0' | ('+'|'-')? '1'..'9' '0'..'9'*;
fragment
EXPONENT : ('e'|'E') ('+'|'-')? ('0'..'9')+ ;
fragment
FLOAT
    :   ('+'|'-')? ('0'..'9')+ '.' ('0'..'9')* EXPONENT?
    |   '.' ('0'..'9')+ EXPONENT?
    |   ('0'..'9')+ EXPONENT
    ;

ID
  :   ('a'..'z'|'A'..'Z')('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'-')*
  ;






definitions : schemaList ruleList EOF;

schemaList : schema *;

schema : (a1=ASTERISK)? pname=ID
	'(' types+=predArg (',' types+=predArg)* ')'
	;

predArg : type_=ID (name=ID)? uni='!'?  ;

ruleList : (mlnRule)*
    ;

mlnRule
    :
    (softRule | hardRule)
    ;

softRule
    :   (du='@')? weight=NUMBER fc=foclause
    |
    (du='@')? warg=ID ':' fc=foclause
    ;



hardRule
    :   fc=foclause PERIOD
    ;

foclause
    :   exq=existQuan?
     (
        ants+=literal
        (',' ants+=literal
        )*
        IMPLIES
     )?
      lits+=literal
     ('v' lits+=literal )*
    ;

existQuan
    :   EXIST
        vs +=ID
        (',' vs+=ID )*;


literal
    :   pref=(PLUS|NOT)?  a = atom
    ;


term
    :
    x = ID
    | d=(NUMBER|STRING)
    ;

atom
    :   pred=ID '('
    terms+=term
    (','
    terms+=term
    )* ')'
    ;

queryList : query+ EOF
  ;

query  :
  a=atom
  ;


evidenceList : evidence+ EOF;


evidence : prior=NUMBER? perf=NOT? a=atom;
