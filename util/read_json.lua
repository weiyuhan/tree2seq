loacl cjson = require "cjson"

function treelstm.read_jsons(path)
	local file = io.open(path, 'r')
	local line
	while true do
		line = file:read()
		if line == nil then break end
		local data = cjson.decode(line)
		local ast = data["ast"]
		local nl = data["comment"]
end

function treelstm.read_json_tree(ast)
	
end