consul = "consul.service.consul:8500"
splay  = "15s"

vault {
  address = "http://vault.service.consul:8200"
  renew   = false
  ssl {
    enabled = false
    verify  = false
  }
}

secret {
	path   = "secret/data-science/optimization_engine_research_ui"
	no_prefix = true
}

prefix {
  path = "consul-config/master/dev/optimization_engine_research_ui"
}