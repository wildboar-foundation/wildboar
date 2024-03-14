#############
API Reference
#############

.. toctree::
   :maxdepth: 2

{% for obj in pages|sort(attribute='name') %}
  {% set display = "utils" not in obj.name or obj._should_skip %}
  {% if obj.all is not none %}
    {% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
  {% else %}
    {% set display = False %}
  {% endif %}

  {% if not obj.top_level_object and obj.display and visible_children and display %}
   {{ obj.include_path }}
  {% endif %}
{% endfor %}

