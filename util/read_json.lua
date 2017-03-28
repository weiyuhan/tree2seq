loacl cjson = require "cjson"

function treelstm.read_jsons(path, tree_vocab, nl_vocab)
	local file = io.open(path, 'r')
	local line
	local count = 0
	local trees = {}
	while true do
		line = file:read()
		if line == nil then break end
		local data = cjson.decode(line)
		local ast = data["root"]
		local nl = data["comment"]
		trees[count] = treelstm.read_json_tree(ast, tree_vocab)
		count = count + 1
end

function treelstm.read_json_tree(ast, vocab)
	local root = treelstm.Tree()
	local children = ast["children"]
	for i, p in ipairs(children) do
		root:add_child(treelstm.read_json_tree(p))
	root.num_children = #children
	root.value = vocab:index(ast.type)
	return root
end