" .vimrc

set nu hls et ts=2 sts=2 sw=2 fo=croqtjn tw=70
set autowrite
nnoremap \vv :so .vimrc<cr>
nnoremap \t :w<cr>:call RunScript()<cr>
vert bel terminal

function! GetStandardImports()
  let begin = search('begin standard imports','n')
  let end = search('end standard imports','n')
  return getline(begin,end)
endfunction

function! GetScript()
  let begin = search('^##','nb')
  let end = search('^##','n')
  let this_script = getline(begin,end)
  "return GetStandardImports() + ['print("'.this_script[0].'")'] + this_script
  return GetStandardImports() + this_script
endfunction

function! RunScript()
  call writefile(GetScript(),'script.py')
  !sed '0,/end standard imports/d' script.py
  !python3 script.py
endfunction
