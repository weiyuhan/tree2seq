local cjson = require "cjson"

function treelstm.read_sentences(path)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = tonumber(token)
    end
    sentences[#sentences + 1] = sent
  end
  file:close()
  return sentences
end

function treelstm.read_trees(path)
  local file = io.open(path, 'r')
  local count = 0
  local trees = {}

  while true do
    local line = file:read()
    if line == nil then break end
    local ast = cjson.decode(line)

    count = count + 1
    trees[count] = treelstm.read_tree(ast)
  end
  file:close()
  return trees
end

function treelstm.read_tree(ast)
  local tree = treelstm.Tree()
  tree.value = tonumber(ast["ids"])
  children = ast["children"]
  tree.num_children = #children
  for i, p in ipairs(children) do
    local child = treelstm.read_tree(p)
    tree:add_child(child)
  return root
end


function treelstm.read_dataset(dir, ast_vocab, nl_vocab)
  local dataset = {}
  dataset.ast_vocab = ast_vocab
  dataset.nl_vocab = nl_vocab

  local trees = treelstm.read_trees(dir .. 'dparents.txt', dir .. 'dlabels.txt')
  local sents = treelstm.read_sentences(dir .. 'sents.txt', vocab)

  dataset.trees = trees
  dataset.sents = sents
  return dataset
end
