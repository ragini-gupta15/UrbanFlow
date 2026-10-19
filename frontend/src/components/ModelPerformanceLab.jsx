import "./ModelPerformanceLab.css";
import { Activity, FlaskConical } from "lucide-react";

const experimental = {
  15: {
    weather: { mae: 3.291670, rmse: 5.749414 },
    traffic: { mae: 3.144881, rmse: 5.844688 },
    persistence: { mae: 3.483514, rmse: 6.374143 },
    count: 1243168,
  },
  30: {
    weather: { mae: 3.916554, rmse: 7.045038 },
    traffic: { mae: 3.770267, rmse: 7.215984 },
    persistence: { mae: 4.175889, rmse: 7.983383 },
    count: 1243750,
  },
  45: {
    weather: { mae: 4.450642, rmse: 8.013977 },
    traffic: { mae: 4.276558, rmse: 8.210785 },
    persistence: { mae: 4.778861, rmse: 9.222697 },
    count: 1241932,
  },
  60: {
    weather: { mae: 4.942532, rmse: 8.706656 },
    traffic: { mae: 4.746236, rmse: 8.975672 },
    persistence: { mae: 5.331637, rmse: 10.263740 },
    count: 1241314,
  },
};

const fmt = (value) => value.toFixed(3);
const fmtCount = (value) => new Intl.NumberFormat("en-US").format(value);

export default function ModelPerformanceLab({ horizon = 30 }) {
  const test = experimental[horizon];
  const bestMae = Math.min(test.weather.mae, test.traffic.mae, test.persistence.mae);
  const bestRmse = Math.min(test.weather.rmse, test.traffic.rmse, test.persistence.rmse);

  return (
    <section className="mpl section-card" aria-labelledby="mpl-title">
      <div className="section-card__head">
        <div>
          <span className="section-card__eyebrow">Evaluation workspace · offline results</span>
          <h2 id="mpl-title">Experimental model comparison</h2>
          <p className="mpl-intro">
            Compare verified historical test errors across forecast horizons.
            Lower MAE and RMSE indicate better predictions.
          </p>
        </div>
        <div className="mpl-mark"><FlaskConical size={19} /></div>
      </div>


      <div className="mpl-summary">
        <div>
          <span>Selected horizon</span>
          <strong>{horizon} <small>minutes</small></strong>
        </div>
        <div>
          <span>Matched test observations</span>
          <strong>{fmtCount(test.count)}</strong>
        </div>
        <div>
          <span>Lowest experimental MAE</span>
          <strong>{fmt(bestMae)} <small>mph</small></strong>
        </div>
      </div>

      <div className="mpl-table-wrap">
        <table className="mpl-table">
          <thead>
            <tr>
              <th>Experimental model</th>
              <th>MAE <small>mph</small></th>
              <th>RMSE <small>mph</small></th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Weather + graph-aware", test.weather],
              ["Traffic-only", test.traffic],
              ["Persistence baseline", test.persistence],
            ].map(([name, values]) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td>
                  <span className={values.mae === bestMae ? "mpl-best-value" : ""}>
                    {fmt(values.mae)}
                  </span>
                  {values.mae === bestMae && <small className="mpl-best-label">Best MAE</small>}
                </td>
                <td>
                  <span className={values.rmse === bestRmse ? "mpl-best-value" : ""}>
                    {fmt(values.rmse)}
                  </span>
                  {values.rmse === bestRmse && <small className="mpl-best-label">Best RMSE</small>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mpl-observation">
        <Activity size={17} />
        <p>
          At this horizon, the {test.traffic.mae < test.weather.mae ? "traffic-only model has lower MAE" : "weather-aware model has lower MAE"};
          {" "}{test.weather.rmse < test.traffic.rmse ? "the weather-aware model has lower RMSE" : "the traffic-only model has lower RMSE"}.
          These are separate experimental checkpoints evaluated on matched observations, not proof that weather alone caused the difference.
        </p>
      </div>


    </section>
  );
}
