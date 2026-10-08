Changed
^^^^^^^

* Cached normalized fragment-field surface geometry once per explicit MPM grid
  update and rejected separating field pairs before normal normalization and gap
  evaluation. Contact projection order, timestep, mass and material parameters
  remain unchanged; the optimization targets fragmented cake simulation cost.
