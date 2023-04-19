{% macro auto_summary(objs, title) -%}

.. list-table:: {{ title }}
  :header-rows: 0
  :widths: auto
  :class: autosummary

{% for obj in objs|sort(attribute='name') %}
  * - :py:obj:`~{{ obj.id }}`
    - {{ obj.summary }}
{% endfor %}
{% endmacro %}

{% for obj in pages|sort(attribute='name') %}
  {% set display = "utils" not in obj.name or obj._should_skip %}
  {% if obj.all is not none %}
    {% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
  {% else %}
    {% set display = False %}
  {% endif %}

  {% if obj.display and visible_children and display %}
:py:mod:`{{ obj.id }}`
~~~~~~~~~{{ "~" * obj.id|length }}~
{{ obj.docstring }}

      {% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
      {% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
      {% if visible_classes %}
{{ auto_summary(visible_classes, title="Classes") }}
      {% endif %}

      {% if visible_functions %}
{{ auto_summary(visible_functions, title="Functions") }}
     {% endif %}
   {% endif %}
{% endfor %}