window.ORBIT_BOARD = {
  "protocol": "team-eval-v1",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "honesty": "Task-success intervals include zero. The gain is not statistically significant.",
  "runs": [
    {
      "id": "single_base",
      "label": "Single base",
      "pending": false,
      "n_success": 2,
      "n_tasks": 12,
      "task_success_rate": 0.16666666666666666,
      "schema_valid_rate": 0.75,
      "n_traps": 2,
      "n_correct_refusals": 2,
      "success_interval": {
        "point": 0.16666666666666666,
        "low": 0.0,
        "high": 0.4166666666666667,
        "n": 12,
        "n_resamples": 10000,
        "seed": 0,
        "method": "percentile_bootstrap"
      },
      "schema_interval": {
        "point": 0.75,
        "low": 0.5,
        "high": 1.0,
        "n": 12,
        "n_resamples": 10000,
        "seed": 0,
        "method": "percentile_bootstrap"
      }
    },
    {
      "id": "team_base",
      "label": "Team base",
      "pending": false,
      "n_success": 2,
      "n_tasks": 12,
      "task_success_rate": 0.16666666666666666,
      "schema_valid_rate": 0.6071428571428571,
      "n_traps": 2,
      "n_correct_refusals": 2,
      "success_interval": {
        "point": 0.16666666666666666,
        "low": 0.0,
        "high": 0.4166666666666667,
        "n": 12,
        "n_resamples": 10000,
        "seed": 0,
        "method": "percentile_bootstrap"
      },
      "schema_interval": {
        "point": 0.6071428571428571,
        "low": 0.4446428571428572,
        "high": 0.7791666666666668,
        "n": 12,
        "n_resamples": 10000,
        "seed": 0,
        "method": "percentile_bootstrap"
      }
    },
    {
      "id": "team_lora",
      "label": "Team LoRA",
      "pending": false,
      "n_success": 3,
      "n_tasks": 12,
      "task_success_rate": 0.25,
      "schema_valid_rate": 0.7797619047619048,
      "n_traps": 2,
      "n_correct_refusals": 1,
      "success_interval": {
        "point": 0.25,
        "low": 0.0,
        "high": 0.5,
        "n": 12,
        "n_resamples": 10000,
        "seed": 0,
        "method": "percentile_bootstrap"
      },
      "schema_interval": {
        "point": 0.7797619047619048,
        "low": 0.7055555555555556,
        "high": 0.8234126984126985,
        "n": 12,
        "n_resamples": 10000,
        "seed": 0,
        "method": "percentile_bootstrap"
      }
    },
    {
      "id": "team_distill",
      "label": "Team distill",
      "pending": true,
      "n_success": null,
      "n_tasks": 12,
      "task_success_rate": null,
      "schema_valid_rate": null,
      "n_traps": 2,
      "n_correct_refusals": null,
      "success_interval": null,
      "schema_interval": null
    }
  ],
  "paired": {
    "label": "Team LoRA minus team base",
    "task_success_rate": {
      "point": 0.08333333333333333,
      "low": -0.16666666666666666,
      "high": 0.3333333333333333,
      "n": 12,
      "n_resamples": 10000,
      "seed": 0,
      "alpha": 0.05,
      "method": "paired_percentile_bootstrap"
    },
    "schema_valid_rate": {
      "point": 0.17261904761904764,
      "low": 0.0013888888888889117,
      "high": 0.33888888888888885,
      "n": 12,
      "n_resamples": 10000,
      "seed": 0,
      "alpha": 0.05,
      "method": "paired_percentile_bootstrap"
    },
    "distill": null
  },
  "loss": [
    {
      "step": 10,
      "loss": 1.152004337310791
    },
    {
      "step": 20,
      "loss": 0.4951390266418457
    },
    {
      "step": 30,
      "loss": 0.3131009817123413
    },
    {
      "step": 40,
      "loss": 0.2170185327529907
    },
    {
      "step": 50,
      "loss": 0.21146292686462403
    },
    {
      "step": 60,
      "loss": 0.14217815399169922
    },
    {
      "step": 70,
      "loss": 0.09294049143791198
    },
    {
      "step": 80,
      "loss": 0.12511975765228273
    },
    {
      "step": 90,
      "loss": 0.0799569308757782
    },
    {
      "step": 100,
      "loss": 0.049994167685508725
    },
    {
      "step": 110,
      "loss": 0.058334851264953615
    },
    {
      "step": 120,
      "loss": 0.08363301753997802
    },
    {
      "step": 130,
      "loss": 0.041804739832878114
    },
    {
      "step": 140,
      "loss": 0.08010804057121276
    },
    {
      "step": 150,
      "loss": 0.05106291174888611
    }
  ],
  "traps": [
    {
      "id": "test-011",
      "outcomes": {
        "single_base": true,
        "team_base": true,
        "team_lora": false,
        "team_distill": null
      }
    },
    {
      "id": "test-012",
      "outcomes": {
        "single_base": true,
        "team_base": true,
        "team_lora": true,
        "team_distill": null
      }
    }
  ]
};
