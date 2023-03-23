## NFL tracking data playground
Right now this takes NFL player tracking data (generally grabbed from the [2019](https://github.com/nfl-football-ops/Big-Data-Bowl) and 2023 competitions)
and tries to answer the question "does passing downfield open up space to run?" There are a few main components:

1. Tooling to load and standardize the tracking data itself (e.g., rename columns, standardize column values like team abbreviations, rotating the field
   so all plays go bottom-to-top);
2. Preprocessing the data to annotate each play with the total and recent number of pass attempts and their nature/outcome (e.g., attempted, completed,
   intercepted, QB scramble/sack) and the air distance bucket;
3. Evaluating the space ~0.5-1.0s after snap, when both offense and defense are starting to move to assigned positions rather than reacting to the other
   side. Space evaluation is done using a slightly tweaked version of the [Fernandez and Bornn paper](https://www.researchgate.net/publication/324942294_Wide_Open_Spaces_A_statistical_technique_for_measuring_space_creation_in_professional_soccer)
   from SSAC 2018, with the most valuable space being centered where the ball was snapped, expanding more east-west than north-south;
4. Some tooling to generate basic predictive models using XGBoost, StatsModels, and Keras; and
5. Some tooling to create PNGs and ani-GIFs of the positions and heatmaps.

This is very much rough work and work in progress.
