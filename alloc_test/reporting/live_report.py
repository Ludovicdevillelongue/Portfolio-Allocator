import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objs as go
import webbrowser
from waitress import serve
import time
class LiveDashReport:
    def __init__(self, live_allocation_runner, port, update_interval=60000):
        """
        Initialize the DashReport to fetch live metrics every 60 seconds.
        :param live_allocation_runner: Instance of LiveAllocationRunner to fetch live metrics.
        :param port: Port to run the Dash server on.
        :param update_interval: Time interval in milliseconds for updating the dashboard (default: 60000 ms).
        """
        self.live_allocation_runner = live_allocation_runner
        self.port = port
        self.update_interval = update_interval  # Interval for updates (60 seconds by default)
        self.portfolio_metrics = self.live_allocation_runner.get_live_metrics()
        self.dynamic_metrics = ['rolling_sharpe', 'rolling_beta', 'portfolio_values', 'portfolio_cumulative_returns', 'portfolio_pnl',
                                'portfolio_cum_pnl', 'cash']
        self.static_metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'calmar_ratio', 'max_drawdown',
                               'omega_ratio', 'sortino_ratio', 'tail_ratio', 'daily_var']
        self.asset_metrics = ['weights', 'asset_prices', 'positions', 'market_values', 'asset_pnl', 'asset_cum_pnl',
                              'asset_cumulative_returns']
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()



    def _prepare_data_for_table(self):
        # Retrieve best_opti_algo and best_params from the last entry of strategy_history
        strategy_history = self.portfolio_metrics['strategy_history']
        selected_strategy=strategy_history.iloc[-1]['strategy_name']
        best_opti_algo = strategy_history.iloc[-1]['best_opti_algo']
        best_params = strategy_history.iloc[-1]['best_params']

        # Prepare the data for the static table from portfolio metrics
        metrics_data = {
            k: round(v, 2) if isinstance(v, (float, int)) else v
            for k, v in self.portfolio_metrics.items() if k in self.static_metrics
        }

        # Add best_opti_algo and best_params to the dictionary*
        metrics_data['actual_strategy'] = selected_strategy
        metrics_data['best_opti_algo'] = best_opti_algo
        metrics_data['best_params'] = best_params

        return [metrics_data]

    def _setup_layout(self):
        # Define the layout for a single Portfolio
        self.app.layout = html.Div([
            html.H1('Portfolio Static Performance Metrics'),
            dash_table.DataTable(
                id='performance-metrics-table',
                columns=[{"name": i, "id": i} for i in self._prepare_data_for_table()[0].keys()],
                data=self._prepare_data_for_table(),
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center'}
            ),

            html.H1('Portfolio Dynamic Performance Metrics'),
            *[dcc.Graph(id=f'graph-{metric}', figure={
                'data': [go.Scatter(x=self.portfolio_metrics[metric].index, y=self.portfolio_metrics[metric], mode='lines', name=metric)],
                'layout': go.Layout(
                    title=f'{metric.capitalize()}',
                        xaxis={
                            'title': 'Date',
                            'rangeselector': dict(
                                buttons=list([
                                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                                    dict(count=1, label="1D", step="day", stepmode="backward"),
                                    dict(count=7, label="1W", step="day", stepmode="backward"),
                                    dict(count=1, label="1M", step="month", stepmode="backward"),
                                    dict(count=3, label="3M", step="month", stepmode="backward"),
                                    dict(count=6, label="6M", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            'rangeslider': {'visible': True}
                        },
                    yaxis={'title': metric.replace("_", " ").title()}
                )
            }) for metric in self.dynamic_metrics],

            html.H1('Asset-Level Metrics'),
            *[dcc.Graph(id=f'asset-{metric}-graph') for metric in
              self.asset_metrics],

            # Interval component to trigger updates every 60 seconds
            dcc.Interval(id='interval-component', interval=self.update_interval, n_intervals=0)
        ])

    def _setup_callbacks(self):
        # Set up callback for updating live metrics every 60 seconds
        @self.app.callback(
            [
                Output('performance-metrics-table', 'data'),
                *[Output(f'graph-{metric}', 'figure') for metric in self.dynamic_metrics],
                *[Output(f'asset-{metric}-graph', 'figure') for metric in
                  self.asset_metrics],
            ],
            [Input('interval-component', 'n_intervals')]
        )
        def update_live_data(n_intervals):
            if n_intervals == 0:
                raise dash.exceptions.PreventUpdate

            # Fetch the latest live portfolio metrics from the live allocation runner
            self.portfolio_metrics = self.live_allocation_runner.get_live_metrics()

            # Prepare the new data for the static table
            performance_data = self._prepare_data_for_table()

            # Update rolling metric graphs
            rolling_metric_figures = [
                {
                    'data': [go.Scatter(x=self.portfolio_metrics[metric].index, y=self.portfolio_metrics[metric], mode='lines', name=metric)],
                    'layout': go.Layout(
                        title=f'{metric.capitalize()}',
                        xaxis={
                            'title': 'Date',
                            'rangeselector': dict(
                                buttons=list([
                                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                                    dict(count=1, label="1D", step="day", stepmode="backward"),
                                    dict(count=7, label="1W", step="day", stepmode="backward"),
                                    dict(count=1, label="1M", step="month", stepmode="backward"),
                                    dict(count=3, label="3M", step="month", stepmode="backward"),
                                    dict(count=6, label="6M", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            'rangeslider': {'visible': True}
                        },
                        yaxis={'title': metric.replace("_", " ").title()}
                    )
                }
                for metric in self.dynamic_metrics
            ]

            # Update asset-level graphs
            asset_metric_figures = [
                {
                    'data': [
                        go.Scatter(x=metric.index, y=metric[symbol], mode='lines', name=symbol)
                        for symbol in metric.columns
                    ],
                    'layout': go.Layout(
                        title=f'{metric_name.capitalize()}',
                        xaxis={
                            'title': 'Date',
                            'rangeselector': dict(
                                buttons=list([
                                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                                    dict(count=1, label="1D", step="day", stepmode="backward"),
                                    dict(count=7, label="1W", step="day", stepmode="backward"),
                                    dict(count=1, label="1M", step="month", stepmode="backward"),
                                    dict(count=3, label="3M", step="month", stepmode="backward"),
                                    dict(count=6, label="6M", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            'rangeslider': {'visible': True}
                        },
                        yaxis={'title': metric_name.replace("_", " ").title()}
                    )
                }
                for metric_name, metric in self.portfolio_metrics.items()
                if metric_name in self.asset_metrics
            ]

            return [performance_data] + rolling_metric_figures + asset_metric_figures

    def run_server(self):
        # Open the browser and run the Dash server
        webbrowser.open(f"http://127.0.0.1:{self.port}")
        serve(self.app.server, host='0.0.0.0', port=self.port)

